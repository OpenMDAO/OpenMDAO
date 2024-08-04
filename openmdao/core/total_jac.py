"""
Helper class for total jacobian computation.
"""
import sys
import time
import pprint
from contextlib import contextmanager
from collections import defaultdict
from itertools import repeat
from copy import deepcopy

import numpy as np

from openmdao.core.constants import INT_DTYPE
from openmdao.utils.mpi import MPI, check_mpi_env
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.relevance import get_relevance


use_mpi = check_mpi_env()
if use_mpi is not False:
    try:
        from petsc4py import PETSc
    except ImportError:
        PETSc = None
        if use_mpi is True:
            raise ImportError("Importing petsc4py failed and OPENMDAO_USE_MPI is true.")
elif use_mpi is False:
    PETSc = None

_directional_rng = np.random.default_rng(99)


class _TotalJacInfo(object):
    """
    Object to manage computation of total derivatives.

    Attributes
    ----------
    comm : MPI.Comm or <FakeComm>
        The global communicator.
    debug_print : bool
        When True, print out debug and timing information for each derivative solved.
    has_lin_cons : bool
        If True, this total jacobian contains linear constraints.
    idx_iter_dict : dict
        A dict containing an entry for each outer iteration of the total jacobian computation.
    J : ndarray
        The dense array form of the total jacobian.
    J_dict : dict
        Nested or flat dict with views of the jacobian.
    J_final : ndarray or dict
        If return_format is 'array', Jfinal is J.  Otherwise it's either a nested dict (if
        return_format is 'dict') or a flat dict (return_format 'flat_dict') with views into
        the array jacobian.
    lin_sol_cache : dict
        Dict of indices keyed to solution vectors.
    mode : str
        If 'fwd' compute deriv in forward mode, else if 'rev', reverse (adjoint) mode.
    model : <System>
        The top level System of the System tree.
    output_tuple : tuple of str
        Tuple of names of output variables for this total jacobian.  In fwd mode, outputs
        are responses.  In rev mode, outputs are design variables.
    output_vec : dict of Vector.
        Designated linear output vectors based on value of mode ('fwd' or 'rev').
    owning_ranks : dict
        Map of absolute var name to the MPI process that owns it.
    par_deriv : dict
        Cache containing names of desvars or responses for each parallel derivative color.
    par_deriv_printnames : dict
        Companion to par_deriv cache with auto_ivc names mapped to their promoted inputs.
        This is used for debug printing.
    return_format : str
        Indicates the desired return format of the total jacobian. Can have value of
        'array', 'dict', or 'flat_dict'.
    simul_coloring : Coloring or None
        Contains all data necessary to simultaneously solve for groups of total derivatives.
    _dist_driver_vars : dict
        Dict of constraints that are distributed outputs. Key is 'user' variable name, typically
        promoted name or an alias, and values are (local indices, local sizes).
    in_idx_map : dict
        Mapping of jacobian row/col index to a tuple of the form
        (relevant_systems, cache_linear_solutions_flag, voi name)
    directional : bool
        If True, perform a single directional derivative.
    relevance : dict
        Dict of relevance dictionaries for each var of interest.
    """

    def __init__(self, problem, of, wrt, return_format, approx=False,
                 debug_print=False, driver_scaling=True, get_remote=True, directional=False,
                 coloring_info=None, driver=None):
        """
        Initialize object.

        Parameters
        ----------
        problem : <Problem>
            Reference to that Problem object that contains this _TotalJacInfo.
        of : iter of str
            Response names.
        wrt : iter of str
            Design variable names.
        return_format : str
            Indicates the desired return format of the total jacobian. Can have value of
            'array', 'dict', or 'flat_dict'.
        approx : bool
            If True, the object will compute approx total jacobians.
        debug_print : bool
            Set to True to print out debug and timing information for each derivative solved.
        driver_scaling : bool
            If True (default), scale derivative values by the quantities specified when the desvars
            and responses were added. If False, leave them unscaled.
        get_remote : bool
            Whether to get remote variables if using MPI.
        directional : bool
            If True, perform a single directional derivative.
        coloring_info : ColoringMeta, None, or False
            If None, use driver coloring if it exists.  If False, do no coloring. Otherwise, either
            use or generate a new coloring based on the state of the coloring_info object.
        driver : <Driver>, None, or False
            The driver that owns the total jacobian.  If None, use the driver from the problem.
            If False, this total jacobian will be computed directly by the problem.
        """
        if driver is None:
            driver = problem.driver
        self.model = model = problem.model

        self.comm = problem.comm
        self._orig_mode = problem._orig_mode
        self.has_scaling = driver and driver._has_scaling and driver_scaling
        self.return_format = return_format
        self.lin_sol_cache = {}
        self.debug_print = debug_print
        self.par_deriv_printnames = {}
        self.get_remote = get_remote
        self.directional = directional
        self.initialize = True
        self.approx = approx
        self.coloring_info = coloring_info
        try:
            self._linear_only_dvs = set(driver._lin_dvs).difference(driver._nl_dvs)
        except AttributeError:
            self._linear_only_dvs = set()

        orig_of = of
        orig_wrt = wrt

        if not model._use_derivatives:
            raise RuntimeError("Derivative support has been turned off but compute_totals "
                               "was called.")

        of_metadata, wrt_metadata, has_custom_derivs = model._get_totals_metadata(driver, of, wrt)

        ofsize = sum(meta['global_size'] for meta in of_metadata.values())
        wrtsize = sum(meta['global_size'] for meta in wrt_metadata.values())

        all_lin_cons = True
        has_lin_cons = False
        if driver and driver.supports['linear_constraints']:
            for meta in of_metadata.values():
                if 'linear' in meta and meta['linear']:
                    has_lin_cons = True
                    continue

                all_lin_cons = False
        else:
            all_lin_cons = False

        if self._orig_mode == 'auto':
            if has_lin_cons:
                self.mode = model._mode
            elif ofsize >= wrtsize:
                self.mode = 'fwd'
            else:
                self.mode = 'rev'
        else:
            self.mode = self._orig_mode

        self.input_meta = {'fwd': wrt_metadata, 'rev': of_metadata}
        self.output_meta = {'fwd': of_metadata, 'rev': wrt_metadata}
        self.input_vec = {'fwd': model._dresiduals, 'rev': model._doutputs}
        self.output_vec = {'fwd': model._doutputs, 'rev': model._dresiduals}
        self._dist_driver_vars = driver._dist_driver_vars if driver else {}

        all_abs2meta_out = model._var_allprocs_abs2meta['output']

        self.has_lin_cons = has_lin_cons
        self.dist_input_range_map = {}

        self.simul_coloring = None

        self.relevance = get_relevance(model, of_metadata, wrt_metadata)

        if not all_lin_cons:
            self._check_discrete_dependence()

        if approx:
            coloring_mod._initialize_model_approx(model, driver, of_metadata, wrt_metadata)
            modes = [self.mode]
        else:
            if not has_lin_cons:
                if driver and ((orig_of is None and orig_wrt is None) or not has_custom_derivs):
                    # we're using driver ofs/wrts
                    if coloring_info is None:
                        self.coloring_info = coloring_info = driver._coloring_info

                do_coloring = coloring_info and \
                    coloring_info.do_compute_coloring() and (coloring_info.dynamic) \
                    and not problem._computing_coloring

                if do_coloring:
                    run_model = coloring_info.run_model if 'run_model' in coloring_info else None

                    coloring_info.coloring = problem.get_total_coloring(coloring_info,
                                                                        of=of_metadata,
                                                                        wrt=wrt_metadata,
                                                                        run_model=run_model)

                if coloring_info:
                    self.simul_coloring = coloring_info.coloring

            if self.simul_coloring is None:
                modes = [self.mode]
            else:
                modes = self.simul_coloring.modes()

            self.in_idx_map = {}
            self.in_loc_idxs = {}
            self.idx_iter_dict = {}
            self.seeds = {}
            self.nondist_loc_map = {}
            self.loc_jac_idxs = {}
            self.dist_idx_map = {m: None for m in modes}

        self.modes = modes

        self.of_size, _ = self._get_tuple_map(of_metadata, all_abs2meta_out)
        self.wrt_size, self.has_wrt_dist = \
            self._get_tuple_map(wrt_metadata, all_abs2meta_out)

        # always allocate a 2D dense array and we can assign views to dict keys later if
        # return format is 'dict' or 'flat_dict'.
        self.J = J = np.zeros((self.of_size, self.wrt_size))

        # if we have distributed 'wrt' variables in fwd mode we have to broadcast the jac
        # columns from the owner of a given range of dist indices to everyone else.
        if self.get_remote and self.has_wrt_dist and self.comm.size > 1:
            abs2idx = model._var_allprocs_abs2idx
            sizes = model._var_sizes['output']
            # map which indices belong to dist vars and to which rank
            self.dist_input_range_map['fwd'] = dist_map = []
            start = end = 0
            for meta in self.input_meta['fwd'].values():
                src = meta['source']
                slc = meta['jac_slice']
                end += (slc.stop - slc.start)
                if meta['distributed']:
                    # get owning rank for each part of the distrib var
                    varidx = abs2idx[src]
                    distsz = sizes[:, varidx]
                    dstart = dend = start
                    for rank, sz in enumerate(distsz):
                        dend += sz
                        if sz > 0:
                            dist_map.append((dstart, dend, rank))
                        dstart = dend
                start = end

        # create scratch array for jac scatters
        self.jac_scratch = None

        if self.comm.size > 1 and self.get_remote:
            # need 2 scratch vectors of the same size here
            mxsize = 0
            if 'fwd' in modes:
                mxsize = J.shape[0]
            if 'rev' in modes:
                if J.shape[1] > mxsize:
                    mxsize = J.shape[1]
            scratch = [np.zeros(mxsize, dtype=J.dtype)]
            if self.simul_coloring is not None:
                scratch.append(scratch[0].copy())

            self.jac_scratch = {}
            if 'fwd' in modes:
                self.jac_scratch['fwd'] = [scratch[0][:J.shape[0]]]
                if self.simul_coloring is not None:  # when simul coloring, need two scratch arrays
                    self.jac_scratch['fwd'].append(scratch[1][:J.shape[0]])
            if 'rev' in modes:
                self.jac_scratch['rev'] = [scratch[0][:J.shape[1]]]
                if self.simul_coloring is not None:  # when simul coloring, need two scratch arrays
                    self.jac_scratch['rev'].append(scratch[1][:J.shape[1]])

                # create a column mask to zero out contributions to the Allreduce from
                # duplicated vars
                self.rev_allreduce_mask = np.ones(J.shape[1], dtype=bool)

                start = end = 0
                has_dist = False
                for name, meta in wrt_metadata.items():
                    end += meta['size']
                    dist = meta['distributed']
                    has_dist |= dist
                    if not dist and model._owning_rank[meta['source']] != model.comm.rank:
                        self.rev_allreduce_mask[start:end] = False
                    start = end

                # if rev_allreduce_mask isn't all True on all procs, then we need to do an Allreduce
                need_allreduce = not np.all(self.rev_allreduce_mask)
                if not (has_dist or any(model.comm.allgather(need_allreduce))):
                    self.rev_allreduce_mask = None

        if not approx:
            for mode in modes:
                self._create_in_idx_map(mode)

            if directional and 'fwd' not in modes:
                # causes creation of seed for fwd mode that we'll use with fd later
                self._create_in_idx_map('fwd')

            self.sol2jac_map = {}
            for mode in modes:
                self.sol2jac_map[mode] = self._get_sol2jac_map(self.output_meta[mode],
                                                               all_abs2meta_out, mode)
            self.jac_scatters = {}
            self.tgt_petsc = {n: {} for n in modes}
            self.src_petsc = {n: {} for n in modes}
            if 'fwd' in modes:
                self.jac_scatters['fwd'] = self._compute_jac_scatters('fwd', J.shape[0], get_remote)
            if 'rev' in modes:
                self.jac_scatters['rev'] = self._compute_jac_scatters('rev', J.shape[1], get_remote)

        if not self.get_remote:
            for mode in modes:
                # If we're running with only a local total jacobian, then we need to keep
                # track of which rows/cols actually exist in our local jac and what the
                # mapping is between the global row/col index and our local index.
                locs = np.nonzero(self.in_loc_idxs[mode] != -1)[0]
                arr = np.full(self.in_loc_idxs[mode].size, -1.0, dtype=INT_DTYPE)
                arr[locs] = range(locs.size)
                self.loc_jac_idxs[mode] = arr

                # a mapping of which indices correspond to distrib vars so
                # we can exclude them from jac scatters or allreduces
                self.dist_idx_map[mode] = dist_map = np.zeros(arr.size, dtype=bool)
                start = end = 0
                for meta in self.output_meta[mode].values():
                    name = meta['source']
                    end += all_abs2meta_out[name]['size']
                    if all_abs2meta_out[name]['distributed']:
                        dist_map[start:end] = True
                    start = end

        # for dict type return formats, map var names to views of the Jacobian array.
        if return_format == 'array':
            self.J_final = J
            self.J_dict = self._get_dict_J(J, wrt_metadata, of_metadata, 'dict')
        else:
            self.J_final = self.J_dict = self._get_dict_J(J, wrt_metadata, of_metadata,
                                                          return_format)

    def _check_discrete_dependence(self):
        model = self.model
        # raise an exception if we depend on any discrete outputs
        if model._var_allprocs_discrete['output']:
            # discrete_outs at the model level are absolute names
            relevance = model._relevance
            discrete_outs = model._var_allprocs_discrete['output']
            disc_arr = relevance._vars2rel_array(discrete_outs)

            with relevance.all_seeds_active():
                if relevance.any_relevant(discrete_outs):
                    for resp, rmeta in self.output_meta['fwd'].items():
                        for dv, dvmeta in self.input_meta['fwd'].items():
                            relarr = relevance._seed_var_map[dvmeta['source']][rmeta['source']]
                            depdisc = disc_arr & relarr
                            if np.any(depdisc):
                                discnames = relevance.rel_vars_iter(depdisc)
                                raise RuntimeError(f"Total derivative of '{resp}' with respect to "
                                                   f"'{dv}' depends upon discrete output variables "
                                                   f"{sorted(discnames)}.")

    @property
    def msginfo(self):
        """
        Our class name.  For use in error messages/tracers.

        Returns
        -------
        str
            Either our instance pathname or class name.
        """
        return f"<class {type(self).__name__}>"

    def _compute_jac_scatters(self, mode, rowcol_size, get_remote):
        """
        Compute scatter between a given local jacobian row/col to others in other procs.
        """
        model = self.model
        nproc = self.comm.size

        if (((mode == 'fwd' and get_remote) or mode == 'rev') and
                (nproc > 1 or (model._full_comm is not None and model._full_comm.size > 1))):
            myrank = self.comm.rank
            if get_remote:
                myoffset = rowcol_size * myrank
            else:  # rev and not get_remote
                # reduce size of vector by not including distrib vars
                arr = np.ones(rowcol_size, dtype=bool)
                start = end = 0
                for name, _ in self.sol2jac_map['rev'][2]:
                    meta = model._var_abs2meta['output'][name]
                    end += meta['size']
                    if meta['distributed']:
                        arr[start:end] = False
                    start = end
                rowcol_size = np.nonzero(arr)[0].size
                if np.all(arr):
                    arr = slice(None)  # save some memory and avoid array copies later
                self.nondist_loc_map[mode] = arr

                loc_size = np.array([rowcol_size], dtype=INT_DTYPE)
                jac_sizes = np.zeros(nproc, dtype=INT_DTYPE)
                self.comm.Allgather(loc_size, jac_sizes)

                myoffset = np.sum(jac_sizes[:myrank])

            _, _, name2jinds = self.sol2jac_map[mode]

            owns = self.model._owning_rank

            abs2meta_out = self.model._var_allprocs_abs2meta['output']
            loc_abs2meta = self.model._var_abs2meta['output']
            sizes = self.model._var_sizes['output']
            abs2idx = self.model._var_allprocs_abs2idx
            full_j_tgts = []
            full_j_srcs = []

            start = end = 0
            for name, srcinds in name2jinds:
                if name not in abs2idx:
                    continue

                is_dist = abs2meta_out[name]['distributed']

                if name in loc_abs2meta:
                    end += abs2meta_out[name]['size']

                if get_remote and is_dist:
                    myinds = srcinds + myoffset
                    for rank in range(nproc):
                        if rank != myrank:
                            offset = rowcol_size * rank   # J is same size on all procs
                            full_j_srcs.append(myinds)
                            full_j_tgts.append(srcinds + offset)
                elif not self.get_remote and not is_dist:
                    var_idx = abs2idx[name]
                    mysize = sizes[myrank, var_idx]
                    if mysize > 0:
                        srcinds = np.arange(start, end, dtype=INT_DTYPE)
                        myinds = srcinds + myoffset
                        for rank in range(nproc):
                            if rank != myrank and sizes[rank, var_idx] > 0:
                                offset = np.sum(jac_sizes[:rank])
                                full_j_srcs.append(myinds)
                                full_j_tgts.append(srcinds + offset)
                elif owns[name] == myrank:
                    myinds = srcinds + myoffset
                    var_idx = abs2idx[name]
                    for rank in range(nproc):
                        if rank != myrank and sizes[rank, var_idx] == 0:
                            offset = rowcol_size * rank   # J is same size on all procs
                            full_j_srcs.append(myinds)
                            full_j_tgts.append(srcinds + offset)

                if name in loc_abs2meta:
                    start = end

            if full_j_srcs:
                full_src_inds = np.hstack(full_j_srcs)
                full_tgt_inds = np.hstack(full_j_tgts)
            else:
                full_src_inds = np.zeros(0, dtype=INT_DTYPE)
                full_tgt_inds = np.zeros(0, dtype=INT_DTYPE)

            tgt_vec = PETSc.Vec().createWithArray(np.zeros(rowcol_size, dtype=float),
                                                  comm=self.comm)
            self.tgt_petsc[mode] = tgt_vec

            src_vec = PETSc.Vec().createWithArray(np.zeros(rowcol_size, dtype=float),
                                                  comm=self.comm)
            self.src_petsc[mode] = src_vec

            src_indexset = PETSc.IS().createGeneral(full_src_inds, comm=self.comm)
            tgt_indexset = PETSc.IS().createGeneral(full_tgt_inds, comm=self.comm)

            return PETSc.Scatter().create(src_vec, src_indexset, tgt_vec, tgt_indexset)

    def _get_dict_J(self, J, wrt_metadata, of_metadata, return_format):
        """
        Create a dict or flat-dict jacobian that maps to views in the given 2D array jacobian.

        Parameters
        ----------
        J : ndarray
            Array jacobian.
        wrt_metadata : dict
            Dict containing metadata for 'wrt' variables.
        of_metadata : dict
            Dict containing metadata for 'of' variables.
        return_format : str
            Indicates the desired form of the returned jacobian.

        Returns
        -------
        dict
            Dict form of the total jacobian that contains views of the ndarray jacobian.
        """
        J_dict = {}
        get_remote = self.get_remote
        if return_format == 'dict':
            for out, ofmeta in of_metadata.items():
                if not get_remote and ofmeta['remote']:
                    continue
                J_dict[out] = outer = {}
                out_slice = ofmeta['jac_slice']
                for inp, wrtmeta in wrt_metadata.items():
                    if get_remote or not wrtmeta['remote']:
                        outer[inp] = J[out_slice, wrtmeta['jac_slice']]

        elif return_format == 'flat_dict':
            for out, ofmeta in of_metadata.items():
                if not get_remote and ofmeta['remote']:
                    continue
                out_slice = ofmeta['jac_slice']
                for inp, wrtmeta in wrt_metadata.items():
                    if get_remote or not wrtmeta['remote']:
                        J_dict[out, inp] = J[out_slice, wrtmeta['jac_slice']]

        elif return_format == 'flat_dict_structured_key':
            # This format is supported by the recorders (specifically the sql recorder), which use
            # numpy structured arrays.
            for out, ofmeta in of_metadata.items():
                if not get_remote and ofmeta['remote']:
                    continue
                out_slice = ofmeta['jac_slice']
                for inp, wrtmeta in wrt_metadata.items():
                    if get_remote or not wrtmeta['remote']:
                        J_dict[f"{out}!{inp}"] = J[out_slice, wrtmeta['jac_slice']]
        else:
            raise ValueError("'%s' is not a valid jacobian return format." % return_format)

        return J_dict

    def _create_in_idx_map(self, mode):
        """
        Create a list that maps a global index to a name, col/row range, and other data.

        Parameters
        ----------
        mode : str
            Derivative solution direction.
        """
        iproc = self.comm.rank
        model = self.model
        has_par_deriv_color = False
        all_abs2meta_out = model._var_allprocs_abs2meta['output']
        var_sizes = model._var_sizes
        var_offsets = model._get_var_offsets()
        abs2idx = model._var_allprocs_abs2idx
        idx_iter_dict = {}  # a dict of index iterators

        simul_coloring = self.simul_coloring
        seed = []
        fwd = mode == 'fwd'

        loc_idxs = []
        idx_map = []
        start = 0
        end = 0

        for name, meta in self.input_meta[mode].items():
            parallel_deriv_color = None

            source = meta['source']

            in_var_meta = all_abs2meta_out[source]
            dist = in_var_meta['distributed']

            if dist:
                end += meta['global_size']
            else:
                end += meta['size']

            parallel_deriv_color = meta['parallel_deriv_color']
            cache_lin_sol = meta['cache_linear_solution']

            if simul_coloring and parallel_deriv_color:
                raise RuntimeError("Using both simul_coloring and parallel_deriv_color with "
                                   f"variable '{name}' is not supported.")

            if parallel_deriv_color is not None:
                if parallel_deriv_color not in self.par_deriv_printnames:
                    self.par_deriv_printnames[parallel_deriv_color] = []

                self.par_deriv_printnames[parallel_deriv_color].append(name)

            in_idxs = meta['indices'] if 'indices' in meta else None

            if in_idxs is None:
                # if the var is not distributed, global_size == local size
                irange = np.arange(in_var_meta['global_size'], dtype=INT_DTYPE)
            else:
                irange = in_idxs.shaped_array(copy=True)

            in_var_idx = abs2idx[source]
            sizes = var_sizes['output']
            offsets = var_offsets['output']
            gstart = np.sum(sizes[:iproc, in_var_idx])
            gend = gstart + sizes[iproc, in_var_idx]

            # if we're doing parallel deriv coloring, we only want to set the seed on one proc
            # for each var in a given color
            if parallel_deriv_color is not None:
                if fwd:
                    with self.relevance.seeds_active(fwd_seeds=(source,)):
                        relev = self.relevance.relevant_vars(source, 'fwd', inputs=False)
                        for s in self.relevance._all_seed_vars['rev']:
                            if s in relev:
                                break
                        else:
                            relev = set()
                else:
                    with self.relevance.seeds_active(rev_seeds=(source,)):
                        relev = self.relevance.relevant_vars(source, 'rev', inputs=False)
                        for s in self.relevance._all_seed_vars['fwd']:
                            if s in relev:
                                break
                        else:
                            relev = set()
            else:
                relev = None

            if not dist:
                # if the var is not distributed, convert the indices to global.
                # We don't iterate over the full distributed size in this case.
                irange += gstart

            # all local idxs that correspond to vars from other procs will be -1
            # so each entry of loc_i will either contain a valid local index,
            # indicating we should set the local vector entry to 1.0 before running
            # solve_linear, or it will contain -1, indicating we should not set any
            # value before calling solve_linear.
            loc_i = np.full(irange.shape, -1, dtype=INT_DTYPE)
            if gend > gstart and (relev is None or relev):
                loc = np.nonzero(np.logical_and(irange >= gstart, irange < gend))[0]
                if in_idxs is None:
                    if dist:
                        loc_i[loc] = range(0, gend - gstart)
                    else:
                        loc_i[loc] = irange[loc] - gstart
                else:
                    loc_i[loc] = irange[loc]
                    loc_i[loc] -= gstart

                loc_offset = offsets[iproc, in_var_idx] - offsets[iproc, 0]
                loc_i[loc] += loc_offset

            loc_idxs.append(loc_i)

            # We apply a -1 here because the derivative of the output is minus the derivative of
            # the input
            seed.append(np.full(irange.size, -1.0, dtype=float))

            imeta = None
            if parallel_deriv_color:
                has_par_deriv_color = True
                if parallel_deriv_color not in idx_iter_dict:
                    it = self.par_deriv_iter
                    imeta = defaultdict(bool)
                    imeta['par_deriv_color'] = parallel_deriv_color
                    imeta['idx_list'] = [(start, end)]
                    imeta['seed_vars'] = {source}
                    idx_iter_dict[parallel_deriv_color] = (imeta, it)
                else:
                    imeta = idx_iter_dict[parallel_deriv_color][0]
                    if isinstance(imeta['seed_vars'], tuple):
                        imeta['seed_vars'] = set(imeta['seed_vars'])  # change back to a set
                    imeta['idx_list'].append((start, end))
                    imeta['seed_vars'].add(source)
            elif self.directional:
                imeta = defaultdict(bool)
                imeta['idx_list'] = range(start, end)
                imeta['seed_vars'] = {source}
                idx_iter_dict[name] = (imeta, self.directional_iter)
            elif not simul_coloring:  # plain old single index iteration
                imeta = defaultdict(bool)
                imeta['idx_list'] = range(start, end)
                imeta['seed_vars'] = {source}
                idx_iter_dict[name] = (imeta, self.single_index_iter)

            # make the seeds hashable
            if imeta is not None:
                imeta['seed_vars'] = tuple(sorted(imeta['seed_vars']))

            tup = (cache_lin_sol, name, source)

            idx_map.extend([tup] * (end - start))
            start = end

        if has_par_deriv_color:
            _fix_pdc_lengths(idx_iter_dict)

        loc_idxs = np.hstack(loc_idxs)
        seed = np.hstack(seed)

        if simul_coloring:
            simul_color_mode = simul_coloring._fwd if mode == 'fwd' else simul_coloring._rev

        if self.directional:
            seed[:] = _directional_rng.random(seed.size)
            seed *= 2.0
            seed -= 1.0
        elif simul_coloring and simul_color_mode is not None:
            imeta = defaultdict(bool)
            imeta['coloring'] = simul_coloring
            cache = False
            imeta['itermeta'] = itermeta = []
            locs = None
            for ilist in simul_coloring.color_iter(mode):
                all_vois = set()

                for i in ilist:
                    cache_lin_sol, voiname, voisrc = idx_map[i]
                    cache |= cache_lin_sol
                    all_vois.add(voisrc)

                iterdict = defaultdict(bool)

                if len(ilist) > 1:
                    locs = loc_idxs[ilist]
                    active = locs != -1
                    iterdict['local_in_idxs'] = locs[active]
                    iterdict['seeds'] = seed[ilist][active]

                iterdict['cache_lin_solve'] = cache
                iterdict['seed_vars'] = tuple(all_vois)
                itermeta.append(iterdict)

            idx_iter_dict['@simul_coloring'] = (imeta, self.simul_coloring_iter)

        self.in_idx_map[mode] = idx_map
        self.in_loc_idxs[mode] = loc_idxs
        self.idx_iter_dict[mode] = idx_iter_dict
        self.seeds[mode] = seed

    def _get_sol2jac_map(self, vois, allprocs_abs2meta_out, mode):
        """
        Create a dict mapping vecname and direction to an index array into the solution vector.

        Using the index array to pull values from the solution vector will give the values
        in the order needed by the jacobian.

        Parameters
        ----------
        vois : dict
            Mapping of variable of interest (desvar or response) name to its metadata.
        allprocs_abs2meta_out : dict
            Mapping of absolute output name to metadata for that var across all procs.
        mode : str
            Derivative solution direction.

        Returns
        -------
        ndarray
            Indices into the solution vector.
        ndarray
            Indices into a jacobian row or column.
        dict
            Mapping of var name to jacobian row or column indices.
        """
        model = self.model
        fwd = mode == 'fwd'
        myproc = self.comm.rank
        name2jinds = []  # map varname to jac row or col idxs that we must scatter to other procs

        inds = []
        jac_inds = []
        sizes = model._var_sizes['output']
        slices = model._doutputs.get_slice_dict()
        abs2idx = model._var_allprocs_abs2idx
        jstart = jend = 0

        for name, vmeta in vois.items():
            src = vmeta['source']
            indices = vmeta['indices']

            meta = allprocs_abs2meta_out[src]
            sz = vmeta['global_size'] if self.get_remote else vmeta['size']

            if (src in abs2idx and src in slices and (self.get_remote or not vmeta['remote'])):
                var_idx = abs2idx[src]
                slc = slices[src]
                slcsize = slc.stop - slc.start

                if MPI and meta['distributed'] and self.get_remote:
                    if indices is not None:
                        local_idx, sizes_idx, _ = self._dist_driver_vars[name]

                        dist_offset = np.sum(sizes_idx[:myproc])
                        full_inds = np.arange(slc.start, slc.stop, dtype=INT_DTYPE)
                        inds.append(full_inds[local_idx.as_array()])
                        jac_inds.append(jstart + dist_offset +
                                        np.arange(local_idx.indexed_src_size, dtype=INT_DTYPE))
                        name2jinds.append((src, jac_inds[-1]))
                    else:
                        dist_offset = np.sum(sizes[:myproc, var_idx])
                        inds.append(range(slc.start, slc.stop) if slcsize > 0
                                    else np.zeros(0, dtype=INT_DTYPE))
                        jac_inds.append(np.arange(jstart + dist_offset,
                                        jstart + dist_offset + sizes[myproc, var_idx],
                                        dtype=INT_DTYPE))
                        name2jinds.append((src, jac_inds[-1]))
                else:
                    if indices is None:
                        sol_inds = range(slc.start, slc.stop) if slcsize > 0 \
                            else np.zeros(0, dtype=INT_DTYPE)
                    else:
                        sol_inds = np.arange(slc.start, slc.stop, dtype=INT_DTYPE)
                        sol_inds = sol_inds[indices.flat()]
                    inds.append(sol_inds)
                    jac_inds.append(np.arange(jstart, jstart + sz, dtype=INT_DTYPE))
                    if fwd or not self.get_remote:
                        name2jinds.append((src, jac_inds[-1]))

            if self.get_remote or not vmeta['remote']:
                jend += sz
                jstart = jend

        if inds:
            sol_idxs = np.hstack(inds)
            jac_idxs = np.hstack(jac_inds)
        else:
            sol_idxs = np.zeros(0, dtype=INT_DTYPE)
            jac_idxs = np.zeros(0, dtype=INT_DTYPE)

        return sol_idxs, jac_idxs, name2jinds

    def _get_tuple_map(self, vois, abs2meta_out):
        """
        Create a dict that maps var name to metadata tuple.

        The tuple has the form (jacobian row/column slice, indices, distrib)

        Parameters
        ----------
        vois : dict
            Mapping of variable of interest (desvar or response) name to its metadata.
        abs2meta_out : dict
            Mapping of absolute output var name to metadata for that var.

        Returns
        -------
        dict
            Dict of metadata tuples keyed by output name.
        int
            Total number of rows or columns.
        bool
            True if any named variables are distributed.
        """
        start = 0
        end = 0
        get_remote = self.get_remote
        has_dist = False

        for meta in vois.values():
            if not get_remote and meta['remote']:
                continue

            src = meta['source']

            # this 'size'/'global_size' already takes indices into account
            if get_remote and meta['distributed']:
                size = meta['global_size']
            else:
                size = meta['size']

            has_dist |= abs2meta_out[src]['distributed']

            end += size

            meta['jac_slice'] = slice(start, end)

            start = end

        return end, has_dist  # after the loop, end is the total size

    #
    # outer loop iteration functions
    #
    def single_index_iter(self, imeta, mode):
        """
        Iterate over single indices for a single variable.

        Parameters
        ----------
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Yields
        ------
        int
            Current index.
        method
            Input setter method.
        method
            Jac setter method.
        dict or None
            Iteration metadata.
        """
        for i in imeta['idx_list']:
            yield i, self.single_input_setter, self.single_jac_setter, imeta

    def simul_coloring_iter(self, imeta, mode):
        """
        Iterate over index lists for the simul coloring case.

        Parameters
        ----------
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Yields
        ------
        list of int or int
            Current indices or current index.
        method
            Input setter method.
        method
            Jac setter method.
        dict or None
            Iteration metadata.
        """
        coloring = imeta['coloring']
        input_setter = self.simul_coloring_input_setter
        jac_setter = self.simul_coloring_jac_setter

        for color, ilist in enumerate(coloring.color_iter(mode)):
            yield ilist, input_setter, jac_setter, imeta['itermeta'][color]

    def par_deriv_iter(self, imeta, mode):
        """
        Iterate over index lists for the parallel deriv case.

        Parameters
        ----------
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Yields
        ------
        list of int
            Current indices.
        method
            Input setter method.
        method
            Jac setter method.
        """
        idxs = imeta['idx_list']
        for tup in zip(*idxs):
            yield tup, self.par_deriv_input_setter, self.par_deriv_jac_setter, imeta

    def directional_iter(self, imeta, mode):
        """
        Iterate (once) over a directional index list.

        Parameters
        ----------
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Yields
        ------
        list of int
            Current indices.
        method
            Input setter method.
        method
            Jac setter method.
        """
        for ilist, itermeta in zip([imeta['idx_list']], repeat(imeta)):
            yield ilist, self.directional_input_setter, self.directional_jac_setter, itermeta

    def _zero_vecs(self, mode):
        # clean out vectors from last solve
        self.model._doutputs.set_val(0.0)
        self.model._dresiduals.set_val(0.0)
        if mode == 'rev':
            self.model._dinputs.set_val(0.0)

    #
    # input setter functions
    #
    def single_input_setter(self, idx, imeta, mode):
        """
        Set seed into the input vector in the single index case.

        Parameters
        ----------
        idx : int
            Total jacobian row or column index.
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Returns
        -------
        set
            Set of relevant system names.
        tuple of str or None
            'linear' or None if linear solve caching is inactive.
        int or None
            key used for storage of cached linear solve (if active, else None).
        """
        cache_lin_sol, _, _ = self.in_idx_map[mode][idx]

        self._zero_vecs(mode)

        loc_idx = self.in_loc_idxs[mode][idx]
        if loc_idx >= 0:
            self.input_vec[mode].set_val(self.seeds[mode][idx], loc_idx)

        if cache_lin_sol:
            return ('linear',), (idx, mode)
        else:
            return None, None

    def simul_coloring_input_setter(self, inds, itermeta, mode):
        """
        Set -1's into the input vector in the simul coloring case.

        Parameters
        ----------
        inds : list of int
            Total jacobian row or column indices.
        itermeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Returns
        -------
        set
            Set of relevant system names.
        tuple of str or None
            'linear' or None if linear solve caching is inactive.
        int or None
            key used for storage of cached linear solve (if active, else None).
        """
        if len(inds) == 1:
            return self.single_input_setter(inds[0], None, mode)

        self._zero_vecs(mode)

        self.input_vec[mode].set_val(itermeta['seeds'], itermeta['local_in_idxs'])

        if itermeta['cache_lin_solve']:
            return ('linear',), (inds[0], mode)
        else:
            return None, None

    def par_deriv_input_setter(self, inds, imeta, mode):
        """
        Set -1's into the input vector in the parallel derivative case.

        Parameters
        ----------
        inds : tuple of int
            Total jacobian row or column indices.
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Returns
        -------
        set
            Set of relevant system names.
        list of str or None
            List of vec_names or None if linear solve caching is inactive.
        int or None
            key used for storage of cached linear solve (if active, else None).
        """
        vec_names = set()

        for i in inds:
            if self.in_loc_idxs[mode][i] >= 0:
                vnames, _ = self.single_input_setter(i, imeta, mode)
                if vnames is not None:
                    vec_names.add(vnames[0])

        self.model._problem_meta['parallel_deriv_color'] = imeta['par_deriv_color']

        if vec_names:
            return sorted(vec_names), (inds[0], mode)
        else:
            return None, None

    def directional_input_setter(self, inds, itermeta, mode):
        """
        Set random numbers into the input vector in the directional case.

        Parameters
        ----------
        inds : list of int
            Total jacobian row or column indices.
        itermeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Returns
        -------
        set
            Set of relevant system names.
        None
            Not used.
        None
            Not used.
        """
        self._zero_vecs(mode)

        loc_idxs = self.in_loc_idxs[mode][inds]
        loc_idxs = loc_idxs[loc_idxs >= 0]
        if loc_idxs.size > 0:
            self.input_vec[mode].set_val(self.seeds[mode][inds], loc_idxs)

        return None, None

    #
    # Jacobian setter functions
    #
    def simple_single_jac_scatter(self, i, mode):
        """
        Set the appropriate (local) part of the total jacobian for a single input index.

        Parameters
        ----------
        i : int
            Total jacobian row or column index.
        mode : str
            Direction of derivative solution.
        """
        deriv_idxs, jac_idxs, _ = self.sol2jac_map[mode]
        deriv_val = self.output_vec[mode].asarray()

        if not self.get_remote:
            loc_idx = self.loc_jac_idxs[mode][i]
            if loc_idx >= 0:
                i = loc_idx
            else:
                return

        if mode == 'fwd':
            self.J[jac_idxs, i] = deriv_val[deriv_idxs]
        else:  # rev
            self.J[i, jac_idxs] = deriv_val[deriv_idxs]

    def _jac_setter_dist(self, i, mode):
        """
        Scatter the i'th column or allreduce the i'th row of the jacobian.

        Parameters
        ----------
        i : int
            Total jacobian row or column index.
        mode : str
            Direction of derivative solution.
        """
        if mode == 'fwd':
            if self.get_remote:
                if self.jac_scatters[mode] is not None:
                    self.src_petsc[mode].array = self.J[:, i]
                    self.tgt_petsc[mode].array[:] = self.J[:, i]
                    self.jac_scatters[mode].scatter(self.src_petsc[mode], self.tgt_petsc[mode],
                                                    addv=False, mode=False)
                    self.J[:, i] = self.tgt_petsc[mode].array

        else:  # rev
            if self.get_remote and self.rev_allreduce_mask is not None:
                scratch = self.jac_scratch['rev'][0]
                scratch[:] = 0.0
                scratch[self.rev_allreduce_mask] = self.J[i][self.rev_allreduce_mask]
                self.comm.Allreduce(scratch, self.J[i], op=MPI.SUM)

    def single_jac_setter(self, i, mode, meta):
        """
        Set the appropriate part of the total jacobian for a single input index.

        Parameters
        ----------
        i : int
            Total jacobian row or column index.
        mode : str
            Direction of derivative solution.
        meta : dict
            Metadata dict.
        """
        self.simple_single_jac_scatter(i, mode)
        if self.comm.size > 1:
            self._jac_setter_dist(i, mode)

    def par_deriv_jac_setter(self, inds, mode, meta):
        """
        Set the appropriate part of the total jacobian for multiple input indices.

        Parameters
        ----------
        inds : tuple of int
            Total jacobian row or column indices.
        mode : str
            Direction of derivative solution.
        meta : dict
            Metadata dict.
        """
        if self.comm.size > 1:
            for i in inds:
                if self.in_loc_idxs[mode][i] >= 0:
                    self.simple_single_jac_scatter(i, mode)
                    break
            else:
                i = -1
            if mode == 'fwd':
                if i < 0:
                    byrank = self.comm.allgather((i, None))
                else:
                    byrank = self.comm.allgather((i, self.J[:, i]))
                for ind, col in byrank:
                    if col is not None:
                        self.J[:, ind] = col
            else:  # rev
                if i < 0:
                    byrank = self.comm.allgather((i, None))
                else:
                    byrank = self.comm.allgather((i, self.J[i]))
                for ind, row in byrank:
                    if row is not None:
                        self.J[ind, :] = row
        else:
            for i in inds:
                self.simple_single_jac_scatter(i, mode)

    def simul_coloring_jac_setter(self, inds, mode, meta):
        """
        Set the appropriate part of the total jacobian for simul coloring input indices.

        Parameters
        ----------
        inds : list of int
            Total jacobian row or column indices.
        mode : str
            Direction of derivative solution.
        meta : dict
            Metadata dict.
        """
        row_col_map = self.simul_coloring.get_row_col_map(mode)
        fwd = mode == 'fwd'
        dist = self.comm.size > 1

        J = self.J
        deriv_idxs, jac_idxs, _ = self.sol2jac_map[mode]

        deriv_val = self.output_vec[mode].asarray()
        if self.jac_scratch is None:
            reduced_derivs = deriv_val[deriv_idxs]
        else:
            # if simul_coloring is in effect when comm.size > 1, there will be two scratch arrays,
            # so just always use the last one to avoid overwriting any data.
            reduced_derivs = self.jac_scratch[mode][-1]
            reduced_derivs[:] = 0.0
            reduced_derivs[jac_idxs] = deriv_val[deriv_idxs]

        if fwd:
            for i in inds:
                J[row_col_map[i], i] = reduced_derivs[row_col_map[i]]
                if dist:
                    self._jac_setter_dist(i, mode)
        else:  # rev
            for i in inds:
                J[i, row_col_map[i]] = reduced_derivs[row_col_map[i]]
                if dist:
                    self._jac_setter_dist(i, mode)

    def directional_jac_setter(self, inds, mode, meta):
        """
        Set the appropriate part of the total jacobian for directional input indices.

        Parameters
        ----------
        inds : list of int
            Total jacobian row or column indices.
        mode : str
            Direction of derivative solution.
        meta : dict
            Metadata dict.
        """
        fwd = mode == 'fwd'
        dist = self.comm.size > 1

        J = self.J
        deriv_idxs, jac_idxs, _ = self.sol2jac_map[mode]

        deriv_val = self.output_vec[mode].asarray()
        if self.jac_scratch is None:
            reduced_derivs = deriv_val[deriv_idxs]
        else:
            # if 'directional' is in effect when comm.size > 1, there will be two scratch arrays,
            # so just always use the last one to avoid overwriting any data.
            reduced_derivs = self.jac_scratch[mode][-1]
            reduced_derivs[:] = 0.0
            reduced_derivs[jac_idxs] = deriv_val[deriv_idxs]

        if fwd:
            for i in inds:
                J[:, i] = reduced_derivs
                if dist:
                    self._jac_setter_dist(i, mode)
                break  # only need a single col of Jac for directional
        else:  # rev
            for i in inds:
                J[i, :] = reduced_derivs
                if dist:
                    self._jac_setter_dist(i, mode)
                break  # only need a single row of jac for directional

    def compute_totals(self, progress_out_stream=None):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Parameters
        ----------
        progress_out_stream : None or file-like object
            Where to send human readable output. None by default which suppresses the output.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        self.model._recording_iter.push(('_compute_totals', 0))
        self.model._problem_meta['ncompute_totals'] += 1

        if self.approx:
            try:
                return self._compute_totals_approx(progress_out_stream=progress_out_stream)
            finally:
                self.model._recording_iter.pop()

        try:
            debug_print = self.debug_print
            par_print = self.par_deriv_printnames

            has_lin_cons = self.has_lin_cons

            model = self.model
            # Prepare model for calculation by cleaning out the derivatives vectors.
            model._dinputs.set_val(0.0)
            model._doutputs.set_val(0.0)
            model._dresiduals.set_val(0.0)

            # Linearize Model
            model._tot_jac = self

            with self._totjac_context():
                relevance = self.relevance
                with relevance.active(model.linear_solver.use_relevance()):
                    with relevance.all_seeds_active():
                        try:
                            ln_solver = model._linear_solver
                            with model._scaled_context_all():
                                model._linearize(model._assembled_jac,
                                                 sub_do_ln=ln_solver._linearize_children())
                            if ln_solver._assembled_jac is not None and \
                                    ln_solver._assembled_jac._under_complex_step:
                                model.linear_solver._assembled_jac._update(model)
                            ln_solver._linearize()
                        finally:
                            model._tot_jac = None

                self.J[:] = 0.0

                # Main loop over columns (fwd) or rows (rev) of the jacobian
                for mode in self.modes:
                    fwd = mode == 'fwd'
                    for key, idx_info in self.idx_iter_dict[mode].items():
                        imeta, idx_iter = idx_info
                        for inds, input_setter, jac_setter, itermeta in idx_iter(imeta, mode):
                            model._problem_meta['seed_vars'] = itermeta['seed_vars']
                            _, cache_key = input_setter(inds, itermeta, mode)

                            if debug_print:
                                if par_print and key in par_print:
                                    print('Solving color:', key,
                                          '(' + ', '.join([name for name in par_print[key]]) + ')',
                                          flush=True)
                                else:
                                    if key == '@simul_coloring':
                                        print(f'In mode: {mode}, Solving variable(s) using simul '
                                              'coloring:')
                                        for local_ind in imeta['coloring']._local_indices(inds,
                                                                                          mode):
                                            print(f"   {local_ind}", flush=True)
                                    elif self.directional:
                                        print(f"In mode: {mode}.\n, Solving for directional "
                                              f"derivative wrt '{key}'",)
                                    else:
                                        print(f"In mode: {mode}.\n('{key}', [{inds}])", flush=True)

                                t0 = time.perf_counter()

                            if fwd:
                                fwd_seeds = itermeta['seed_vars']
                                rev_seeds = None
                            else:
                                fwd_seeds = None
                                rev_seeds = itermeta['seed_vars']

                            with relevance.seeds_active(fwd_seeds=fwd_seeds, rev_seeds=rev_seeds):
                                # restore old linear solution if cache_linear_solution was set by
                                # the user for any input variables involved in this linear solution.
                                with model._scaled_context_all():
                                    if (cache_key is not None and not has_lin_cons and
                                            self.mode == mode):
                                        self._restore_linear_solution(cache_key, mode)
                                        model._solve_linear(mode)
                                        self._save_linear_solution(cache_key, mode)
                                    else:
                                        model._solve_linear(mode)

                            if debug_print:
                                print(f'Elapsed Time: {time.perf_counter() - t0} secs\n',
                                      flush=True)

                            jac_setter(inds, mode, imeta)

                            # reset any Problem level data for the current iteration
                            self.model._problem_meta['parallel_deriv_color'] = None
                            self.model._problem_meta['seed_vars'] = None

                # Driver scaling.
                if self.has_scaling:
                    self._do_driver_scaling(self.J_dict)

                # if some of the wrt vars are distributed in fwd mode, we bcast from the rank
                # where each part of the distrib var exists
                if self.get_remote and mode == 'fwd' and self.has_wrt_dist:
                    for start, stop, rank in self.dist_input_range_map[mode]:
                        contig = self.J[:, start:stop].copy()
                        model.comm.Bcast(contig, root=rank)
                        self.J[:, start:stop] = contig

                if debug_print:
                    # Debug outputs scaled derivatives.
                    self._print_derivatives()
        finally:
            self.model._recording_iter.pop()

        return self.J_final

    def _compute_totals_approx(self, progress_out_stream=None):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Uses an approximation method, e.g., fd or cs to calculate the derivatives.

        Parameters
        ----------
        progress_out_stream : None or file-like object
            Where to send human readable output. None by default which suppresses the output.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        model = self.model
        return_format = self.return_format
        debug_print = self.debug_print

        # Prepare model for calculation by cleaning out the derivatives vectors.
        model._dinputs.set_val(0.0)
        model._doutputs.set_val(0.0)
        model._dresiduals.set_val(0.0)

        # Solve for derivs with the approximation_scheme.
        # This cuts out the middleman by grabbing the Jacobian directly after linearization.

        t0 = time.perf_counter()

        with self._totjac_context():
            model._tot_jac = self
            try:
                if self.initialize:
                    self.initialize = False

                    # Need this cache cleared because we re-initialize after linear constraints.
                    model._approx_subjac_keys = None

                    if model._approx_schemes:
                        for scheme in model._approx_schemes.values():
                            scheme._reset()
                        method = list(model._approx_schemes)[0]
                        kwargs = model._owns_approx_jac_meta
                        model.approx_totals(method=method, **kwargs)
                        if progress_out_stream is not None:
                            model._approx_schemes[method]._progress_out = progress_out_stream
                    else:
                        model.approx_totals(method='fd')
                        if progress_out_stream is not None:
                            model._approx_schemes['fd']._progress_out = progress_out_stream

                    model._setup_jacobians(recurse=False)
                    model._setup_approx_derivs()
                    if model._coloring_info.coloring is not None:
                        model._coloring_info._update_wrt_matches(model)

                if self.directional:
                    for scheme in model._approx_schemes.values():
                        seeds = {k: -s for k, s in self.seeds.items()}
                        scheme._totals_directions = seeds
                        scheme._totals_directional_mode = self.mode
                else:
                    for scheme in model._approx_schemes.values():
                        scheme._totals_directions = {}
                        scheme._totals_directional_mode = None

                # Linearize Model
                model._linearize(model._assembled_jac,
                                 sub_do_ln=model._linear_solver._linearize_children())

            finally:
                model._tot_jac = None

            totals = self.J_dict
            if debug_print:
                print(f'Elapsed time to approx totals: {time.perf_counter() - t0} secs\n',
                      flush=True)

            # Driver scaling.
            if self.has_scaling:
                self._do_driver_scaling(totals)

            if return_format == 'array':
                totals = self.J  # change back to array version

            if debug_print:
                # Debug outputs scaled derivatives.
                self._print_derivatives()

        return totals

    def _get_zero_inds(self, meta, jac_arr):
        """
        Get zero indices relative to the named variable for jac row/col 'jac_arr'.

        Parameters
        ----------
        meta : dict
            Variable metadata.
        jac_arr : ndarray
            Row or column of jacobian being checked for zero entries. Note that in this
            array, zero entries are True and nonzero ones are False.

        Returns
        -------
        ndarray
            Index array of zero entries.
        """
        inds = meta['indices']   # these must be indices into the flattened var
        shname = 'global_shape' if self.get_remote else 'shape'
        shape = self.model._var_allprocs_abs2meta['output'][meta['source']][shname]
        vslice = jac_arr[meta['jac_slice']]

        if inds is None:
            zero_idxs = np.atleast_1d(vslice.reshape(shape)).nonzero()
        else:
            zero_idxs = np.nonzero(vslice)
            if zero_idxs[0].size == 0:
                return zero_idxs
            varr = np.zeros(shape, dtype=bool)
            varr.flat[inds.as_array()[zero_idxs]] = True
            zero_idxs = np.nonzero(varr)

        return zero_idxs

    def check_total_jac(self, raise_error=True, tol=1e-16):
        """
        Check recently computed totals derivative jacobian for problems.

        Some optimizers can't handle a jacobian when a design variable has no effect on the
        constraints, or a constraint is unaffected by a design variable. This method
        checks for these cases.

        Parameters
        ----------
        tol : double
            Tolerance for the check.
        raise_error : bool
            If True, raise an exception if a zero row or column is found.
        """
        nzrows, nzcols = np.nonzero(np.abs(self.J) > tol)

        # Check for zero rows, which correspond to constraints unaffected by any design vars.
        col = np.ones(self.J.shape[0], dtype=bool)
        col[nzrows] = False  # False here means nonzero
        if np.any(col):  # there's at least 1 row that's zero across all columns
            zero_rows = []
            for n, meta in self.output_meta['fwd'].items():
                zero_idxs = self._get_zero_inds(meta, col)

                if zero_idxs[0].size > 0:
                    if len(zero_idxs) == 1:
                        zero_rows.append((n, list(zero_idxs[0])))
                    else:
                        zero_rows.append((n, list(zip(*zero_idxs))))

            if zero_rows:
                zero_rows = [f"('{n}', inds={idxs})" for n, idxs in zero_rows]
                msg = (f"Constraints or objectives [{', '.join(zero_rows)}] cannot be impacted by "
                       "the design variables of the problem.")
                if raise_error:
                    raise RuntimeError(msg)
                else:
                    issue_warning(msg, category=DerivativesWarning)

        # Check for zero cols, which correspond to design vars that don't affect anything.
        row = np.ones(self.J.shape[1], dtype=bool)
        row[nzcols] = False  # False in this case means nonzero
        if np.any(row):  # there's at least 1 col that's zero across all rows
            zero_cols = []
            for n, meta in self.input_meta['fwd'].items():
                # don't flag zero cols for linear only dvs
                if n in self._linear_only_dvs:
                    continue
                zero_idxs = self._get_zero_inds(meta, row)

                if zero_idxs[0].size > 0:
                    if len(zero_idxs) == 1:
                        zero_cols.append((n, list(zero_idxs[0])))
                    else:
                        zero_cols.append((n, list(zip(*zero_idxs))))

            if zero_cols:
                zero_cols = [f"('{n}', inds={idxs})" for n, idxs in zero_cols]
                msg = (f"Design variables [{', '.join(zero_cols)}] have no impact on the "
                       "constraints or objective.")
                if raise_error:
                    raise RuntimeError(msg)
                else:
                    issue_warning(msg, category=DerivativesWarning)

    def _restore_linear_solution(self, key, mode):
        """
        Restore the previous linear solution.

        Parameters
        ----------
        key : hashable object
            Key to lookup linear solution.
        mode : str
            Direction of derivative solution.
        """
        lin_sol_cache = self.lin_sol_cache
        if key in lin_sol_cache:
            doutputs = self.output_vec[mode]
            doutputs.set_val(lin_sol_cache[key])
        else:
            lin_sol_cache[key] = deepcopy(self.output_vec[mode].asarray())

    def _save_linear_solution(self, key, mode):
        """
        Save the current linear solution.

        Parameters
        ----------
        key : hashable object
            Key to lookup linear solution.
        mode : str
            Direction of derivative solution.
        """
        self.lin_sol_cache[key][:] = self.output_vec[mode].asarray()

    def _do_driver_scaling(self, J):
        """
        Apply scalers to the jacobian if the driver defined any.

        Parameters
        ----------
        J : dict
            Jacobian to be scaled.
        """
        # use promoted names for design vars and responses
        desvars = self.input_meta['fwd']
        responses = self.output_meta['fwd']

        if self.return_format in ('dict', 'array'):
            for prom_out, odict in J.items():
                oscaler = responses[prom_out].get('total_scaler')

                for prom_in, val in odict.items():
                    iscaler = desvars[prom_in].get('total_scaler')

                    # Scale response side
                    if oscaler is not None:
                        val[:] = (oscaler * val.T).T

                    # Scale design var side
                    if iscaler is not None:
                        val *= 1.0 / iscaler

        elif self.return_format == 'flat_dict':
            for tup, val in J.items():
                prom_out, prom_in = tup
                oscaler = responses[prom_out]['total_scaler']
                iscaler = desvars[prom_in]['total_scaler']

                # Scale response side
                if oscaler is not None:
                    val[:] = (oscaler * val.T).T

                # Scale design var side
                if iscaler is not None:
                    val *= 1.0 / iscaler
        else:
            raise RuntimeError("Derivative scaling by the driver only supports 'dict', "
                               "'array' and 'flat_array' formats at present.")

    def _print_derivatives(self):
        """
        Print out the derivatives when debug_print is True.
        """
        if self.return_format == 'dict':
            J_dict = self.J_dict
            for of, wrt_dict in J_dict.items():
                for wrt, J_sub in wrt_dict.items():
                    pprint.pprint({(of, wrt): J_sub})
        else:
            J = self.J
            for of, ofmeta in self.output_meta['fwd'].items():
                if not self.get_remote and ofmeta['remote']:
                    continue
                out_slice = ofmeta['jac_slice']
                for wrt, wrtmeta in self.input_meta['fwd'].items():
                    if self.get_remote or not wrtmeta['remote']:
                        deriv = J[out_slice, wrtmeta['jac_slice']]
                        pprint.pprint({(of, wrt): deriv})

        print('')
        sys.stdout.flush()

    def record_derivatives(self, requester, metadata):
        """
        Record derivatives to the recorder.

        Parameters
        ----------
        requester : <Driver>
            Object requesting derivatives.
        metadata : dict
            Dictionary containing execution metadata.
        """
        self.model._recording_iter.push((requester._get_name(), requester.iter_count))

        try:
            totals = self._get_dict_J(self.J, self.input_meta['fwd'], self.output_meta['fwd'],
                                      'flat_dict_structured_key')
            requester._rec_mgr.record_derivatives(requester, totals, metadata)

        finally:
            self.model._recording_iter.pop()

    def set_col(self, system, icol, column):
        """
        Set the given column of the total jacobian.

        Parameters
        ----------
        system : System
            System that is setting the column. (not used)
        icol : int
            Index of the column.
        column : ndarray
            Array to be copied into the jacobian column.
        """
        self.J[:, icol] = column

    def _get_as_directional(self, mode=None):
        """
        Return a dict jac with of's combined into a single key (rev mode) or wrt's (fwd mode).

        Parameters
        ----------
        mode : str
            Indicates the direction of the derivative computation.  Must be 'fwd' or 'rev'.

        Returns
        -------
        dict
            Total jacobian dict with ((of1, of2, ... of_n), wrt) keys for each 'wrt' (fwd mode), or
            (of, (wrt1, wrt2, ... wrt_n)) keys for each 'of' (rev mode).
        dict
            Dict of the form {'of': {...}, 'wrt': {...}}, where each sub-entry of 'of' is the
            'of' variable name keyed to its corresponding row slice and each sub-entry of 'wrt'
            is the 'wrt' variable name keyed to its corresponding column slice.
        """
        if mode is None:
            mode = self.mode

        # get a nested dict version of J
        Jdict = self._get_dict_J(self.J, self.input_meta['fwd'], self.output_meta['fwd'], 'dict')
        ofsizes = {}
        wrtsizes = {}
        slices = {'of': {}, 'wrt': {}}
        ofstart = ofend = 0
        for i, (of, wrtdict) in enumerate(Jdict.items()):
            wrtstart = wrtend = 0
            for wrt, sjac in wrtdict.items():
                if i == 0:
                    wrtend += sjac.shape[1]
                    wrtsizes[wrt] = sjac.shape[1]
                    slices['wrt'][wrt] = slice(wrtstart, wrtend)
                    wrtstart = wrtend
                else:
                    break

            ofend += sjac.shape[0]
            ofsizes[of] = sjac.shape[0]
            slices['of'][of] = slice(ofstart, ofend)
            ofstart = ofend

        newJ = {}
        if mode == 'fwd':
            oftup = tuple(slices['of'])
            for wrt, slc in slices['wrt'].items():
                newJ[oftup, wrt] = np.atleast_2d(self.J[:, slc])
        else:  # rev
            wrttup = tuple(slices['wrt'])
            for of, slc in slices['of'].items():
                newJ[of, wrttup] = np.atleast_2d(self.J[slc, :])

        return newJ, slices

    @contextmanager
    def _totjac_context(self):
        """
        Context manager to set current relevance for the Problem.
        """
        old_relevance = self.model._problem_meta['relevance']
        old_mode = self.model._problem_meta['mode']
        self.model._problem_meta['relevance'] = self.relevance
        self.model._problem_meta['mode'] = self.mode

        try:
            yield
        finally:
            self.model._problem_meta['relevance'] = old_relevance
            self.model._problem_meta['mode'] = old_mode


def _fix_pdc_lengths(idx_iter_dict):
    """
    Take any parallel_deriv_color entries and make sure their index arrays are the same length.

    Parameters
    ----------
    idx_iter_dict : dict
        Dict of a name/color mapped to indexing information.
    """
    for imeta, _ in idx_iter_dict.values():
        if imeta['par_deriv_color']:
            range_list = imeta['idx_list']
            lens = np.array([end - start for start, end in range_list])
            maxlen = np.max(lens)
            diffs = lens - maxlen
            if np.any(diffs):
                for i, diff in enumerate(diffs):
                    start, end = range_list[i]
                    if diff < 0:
                        range_list[i] = np.empty(maxlen, dtype=INT_DTYPE)
                        range_list[i][:end - start] = np.arange(start, end, dtype=INT_DTYPE)
                        range_list[i][end - start:] = range_list[i][end - start - 1]
                    else:
                        range_list[i] = np.arange(start, end, dtype=INT_DTYPE)
            else:
                # just convert all (start, end) tuples to ranges
                for i, (start, end) in enumerate(range_list):
                    range_list[i] = range(start, end)
