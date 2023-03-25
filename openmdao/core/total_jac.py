"""
Helper class for total jacobian computation.
"""
from collections import defaultdict
from itertools import chain
from copy import deepcopy
import pprint
import sys
import time

import numpy as np

from openmdao.core.constants import INT_DTYPE
from openmdao.utils.general_utils import ContainsAll, _src_or_alias_dict, _src_or_alias_name

from openmdao.utils.mpi import MPI, check_mpi_env
from openmdao.utils.coloring import _initialize_model_approx, Coloring
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning


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

_contains_all = ContainsAll()


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
    of_meta : dict
        Map of absolute output 'of' var name to tuples of the form
        (row/column slice, indices, distrib).
    wrt_meta : dict
        Map of absolute output 'wrt' var name to tuples of the form
        (row/column slice, indices, distrib).
    ivc_print_names :dict
        Dictionary that maps auto_ivc names back to their promoted input names.
    output_list : list of str
        List of names of output variables for this total jacobian.  In fwd mode, outputs
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
        Dict of constraints that are distributed outputs. Key is abs variable name, values are
        (local indices, local sizes).
    in_idx_map : dict
        Mapping of jacobian row/col index to a tuple of the form
        (ndups, relevant_systems, cache_linear_solutions_flag)
    total_relevant_systems : set
        The set of names of all systems relevant to the computation of the total derivatives.
    """

    def __init__(self, problem, of, wrt, use_abs_names, return_format, approx=False,
                 debug_print=False, driver_scaling=True, get_remote=True):
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
        use_abs_names : bool
            If True, names in of and wrt are absolute names.
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
        """
        driver = problem.driver
        prom2abs = problem.model._var_allprocs_prom2abs_list['output']
        prom2abs_in = problem.model._var_allprocs_prom2abs_list['input']
        conns = problem.model._conn_global_abs_in2out

        self.model = model = problem.model
        self.comm = problem.comm
        self.mode = problem._mode
        self.owning_ranks = problem.model._owning_rank
        self.has_scaling = driver._has_scaling and driver_scaling
        self.return_format = return_format
        self.lin_sol_cache = {}
        self.debug_print = debug_print
        self.par_deriv_printnames = {}
        self.get_remote = get_remote

        if isinstance(wrt, str):
            wrt = [wrt]

        if isinstance(of, str):
            of = [of]

        # convert designvar and response dicts to use src or alias names
        # keys will all be absolute names or aliases after conversion
        design_vars = _src_or_alias_dict(driver._designvars)
        self.responses = responses = _src_or_alias_dict(driver._responses)

        if not model._use_derivatives:
            raise RuntimeError("Derivative support has been turned off but compute_totals "
                               "was called.")

        driver_wrt = list(driver._designvars)
        driver_of = driver._get_ordered_nl_responses()

        # In normal use, of and wrt always contain variable names. However, there are unit tests
        # that don't specify them, so we need these here.
        if wrt is None:
            wrt = driver_wrt
        if of is None:
            of = driver_of

        # Convert 'wrt' names from promoted to absolute
        prom_wrt = wrt
        wrt = []
        self.ivc_print_names = {}
        for name in prom_wrt:
            if name in prom2abs:
                wrt_name = prom2abs[name][0]
            elif name in prom2abs_in:
                in_abs = prom2abs_in[name][0]
                wrt_name = conns[in_abs]
                self.ivc_print_names[wrt_name] = name
            else:
                wrt_name = name
            wrt.append(wrt_name)

        # Convert 'of' names from promoted to absolute (or alias)
        prom_of = of
        of = []
        src_of = []
        for name in prom_of:  # these names could be aliases
            if name in prom2abs:
                of_name = prom2abs[name][0]
            elif name in prom2abs_in:
                # An auto_ivc design var can be used as a response too.
                in_abs = prom2abs_in[name][0]
                of_name = conns[in_abs]
                self.ivc_print_names[of_name] = name
            else:
                of_name = name
            of.append(of_name)
            if name in responses:
                src_of.append(responses[name]['source'])
            else:
                src_of.append(of_name)

        if not get_remote and self.comm.size > 1:
            self.remote_vois = frozenset(n for n in chain(src_of, wrt)
                                         if n not in model._var_abs2meta['output'])
        else:
            self.remote_vois = frozenset()

        # raise an exception if we depend on any discrete outputs
        if model._var_allprocs_discrete['output']:
            discrete_outs = set(model._var_allprocs_discrete['output'])
            inps = of if self.mode == 'rev' else wrt

            for inp in inps:
                inter = discrete_outs.intersection(model._relevant[inp]['@all'][0]['output'])
                if inter:
                    kind = 'of' if self.mode == 'rev' else 'with respect to'
                    raise RuntimeError("Total derivative %s '%s' depends upon "
                                       "discrete output variables %s." %
                                       (kind, inp, sorted(inter)))

        self.of = of
        self.wrt = wrt
        self.prom_of = prom_of
        self.prom_wrt = prom_wrt

        self.input_list = {'fwd': wrt, 'rev': of}
        self.output_list = {'fwd': of, 'rev': wrt}
        self.input_meta = {'fwd': design_vars, 'rev': responses}
        self.output_meta = {'fwd': responses, 'rev': design_vars}
        self.input_vec = {
            'fwd': model._dresiduals,
            'rev': model._doutputs
        }
        self.output_vec = {
            'fwd': model._doutputs,
            'rev': model._dresiduals
        }
        self._dist_driver_vars = driver._dist_driver_vars

        abs2meta_out = model._var_allprocs_abs2meta['output']

        constraints = driver._cons

        for name in prom_of:
            if name in constraints and constraints[name]['linear']:
                has_lin_cons = True
                self.simul_coloring = None
                break
        else:
            has_lin_cons = False

        self.has_lin_cons = has_lin_cons
        self.dist_input_range_map = {}
        self.has_input_dist = {}
        self.has_output_dist = {}

        self.total_relevant_systems = set()

        if approx:
            _initialize_model_approx(model, driver, self.of, self.wrt)
            modes = ['fwd']
        else:
            if not has_lin_cons:
                self.simul_coloring = driver._coloring_info['coloring']

                # if we don't get wrt and of from driver, turn off coloring
                if self.simul_coloring is not None and \
                   (prom_wrt != driver_wrt or prom_of != driver_of):
                    msg = ("compute_totals called using a different list of design vars and/or "
                           "responses than those used to define coloring, so coloring will "
                           "be turned off.\ncoloring design vars: %s, current design vars: "
                           "%s\ncoloring responses: %s, current responses: %s." %
                           (driver_wrt, wrt, driver_of, of))
                    issue_warning(msg, category=DerivativesWarning)
                    self.simul_coloring = None

            if not isinstance(self.simul_coloring, Coloring):
                self.simul_coloring = None

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

        self.of_meta, self.of_size, has_of_dist = \
            self._get_tuple_map(of, responses, abs2meta_out)
        self.has_input_dist['rev'] = self.has_output_dist['fwd'] = has_of_dist
        self.wrt_meta, self.wrt_size, has_wrt_dist = \
            self._get_tuple_map(wrt, design_vars, abs2meta_out)
        self.has_input_dist['fwd'] = self.has_output_dist['rev'] = has_wrt_dist

        # always allocate a 2D dense array and we can assign views to dict keys later if
        # return format is 'dict' or 'flat_dict'.
        self.J = J = np.zeros((self.of_size, self.wrt_size))

        if self.get_remote:
            # if we have distributed 'wrt' variables in fwd mode we have to broadcast the jac
            # columns from the owner of a given range of dist indices to everyone else.
            if has_wrt_dist and self.comm.size > 1:
                abs2idx = model._var_allprocs_abs2idx
                sizes = model._var_sizes['output']
                meta = self.wrt_meta
                # map which indices belong to dist vars and to which rank
                self.dist_input_range_map['fwd'] = dist_map = []
                start = end = 0
                for name in self.input_list['fwd']:
                    slc, _, is_dist = meta[name]
                    end += (slc.stop - slc.start)
                    if is_dist:
                        # get owning rank for each part of the distrib var
                        varidx = abs2idx[name]
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
        self.jac_dist_col_mask = None

        if self.comm.size > 1 and self.get_remote:
            # need 2 scratch vectors of the same size here
            scratch = np.zeros(max(J.shape), dtype=J.dtype)
            scratch2 = scratch.copy()
            self.jac_scratch = {}
            if 'fwd' in modes:
                self.jac_scratch['fwd'] = (scratch[:J.shape[0]], scratch2[:J.shape[0]])
            if 'rev' in modes:
                self.jac_scratch['rev'] = (scratch[:J.shape[1]], scratch2[:J.shape[1]])
                if self.has_output_dist['rev']:
                    sizes = model._var_sizes['output']
                    abs2idx = model._var_allprocs_abs2idx
                    self.jac_dist_col_mask = mask = np.ones(J.shape[1], dtype=bool)
                    start = end = 0
                    for name in self.wrt:
                        meta = abs2meta_out[name]
                        end += meta['global_size']
                        if meta['distributed']:
                            # see if we have an odd dist var like some auto_ivcs connected to
                            # remote vars, which are zero everywhere except for one proc
                            sz = sizes[:, abs2idx[name]]
                            if np.count_nonzero(sz) > 1:
                                mask[start:end] = False
                        start = end

        if not approx:
            for mode in modes:
                self._create_in_idx_map(mode)

            self.sol2jac_map = {}
            for mode in modes:
                self.sol2jac_map[mode] = self._get_sol2jac_map(self.output_list[mode],
                                                               self.output_meta[mode],
                                                               abs2meta_out, mode)
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
                arr[locs] = np.arange(locs.size, dtype=INT_DTYPE)
                self.loc_jac_idxs[mode] = arr

                # a mapping of which indices correspond to distrib vars so
                # we can exclude them from jac scatters or allreduces
                self.dist_idx_map[mode] = dist_map = np.zeros(arr.size, dtype=bool)
                start = end = 0
                for name in self.output_list[mode]:
                    end += abs2meta_out[name]['size']
                    if abs2meta_out[name]['distributed']:
                        dist_map[start:end] = True
                    start = end

        # for dict type return formats, map var names to views of the Jacobian array.
        if return_format == 'array':
            self.J_final = J
            self.J_dict = self._get_dict_J(J, wrt, prom_wrt, of, prom_of,
                                           self.wrt_meta, self.of_meta, 'dict')
        else:
            self.J_final = self.J_dict = self._get_dict_J(J, wrt, prom_wrt, of, prom_of,
                                                          self.wrt_meta, self.of_meta,
                                                          return_format)

        if self.has_scaling:
            self.prom_design_vars = {prom_wrt[i]: design_vars[dv] for i, dv in enumerate(wrt)}
            self.prom_responses = {prom_of[i]: responses[r] for i, r in enumerate(of)}

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
            elif not get_remote:  # rev and not get_remote
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
            loc_abs = self.model._var_abs2meta['output']
            sizes = self.model._var_sizes['output']
            abs2idx = self.model._var_allprocs_abs2idx
            full_j_tgts = []
            full_j_srcs = []

            start = end = 0
            for name, srcinds in name2jinds:
                if name not in abs2idx:
                    continue

                is_dist = abs2meta_out[name]['distributed']

                if name in loc_abs:
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

                if name in loc_abs:
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

    def _get_dict_J(self, J, wrt, prom_wrt, of, prom_of, wrt_meta, of_meta, return_format):
        """
        Create a dict or flat-dict jacobian that maps to views in the given 2D array jacobian.

        Parameters
        ----------
        J : ndarray
            Array jacobian.
        wrt : iter of str
            Absolute names of 'with respect to' vars.
        prom_wrt : iter of str
            Promoted names of 'with respect to' vars.
        of : iter of str
            Absolute names of output vars.
        prom_of : iter of str
            Promoted names of output vars.
        wrt_meta : dict
            Dict mapping absolute 'with respect to' name to array jacobian slice, indices,
            and distrib.
        of_meta : dict
            Dict mapping absolute output name to array jacobian slice, indices, and distrib.
        return_format : str
            Indicates the desired form of the returned jacobian.

        Returns
        -------
        dict
            Dict form of the total jacobian that contains views of the ndarray jacobian.
        """
        J_dict = {}
        if return_format == 'dict':
            for i, out in enumerate(of):
                if out in self.remote_vois:
                    continue
                J_dict[prom_of[i]] = outer = {}
                out_slice = of_meta[out][0]
                for j, inp in enumerate(wrt):
                    if inp not in self.remote_vois:
                        outer[prom_wrt[j]] = J[out_slice, wrt_meta[inp][0]]
        elif return_format == 'flat_dict':
            for i, out in enumerate(of):
                if out in self.remote_vois:
                    continue
                out_slice = of_meta[out][0]
                for j, inp in enumerate(wrt):
                    if inp not in self.remote_vois:
                        J_dict[prom_of[i], prom_wrt[j]] = J[out_slice, wrt_meta[inp][0]]
        elif return_format == 'flat_dict_structured_key':
            # This format is supported by the recorders (specifically the sql recorder), which use
            # numpy structured arrays.
            for i, out in enumerate(of):
                if out in self.remote_vois:
                    continue
                out_slice = of_meta[out][0]
                for j, inp in enumerate(wrt):
                    if inp not in self.remote_vois:
                        key = '!'.join((prom_of[i], prom_wrt[j]))
                        J_dict[key] = J[out_slice, wrt_meta[inp][0]]
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
        relevant = model._relevant
        has_par_deriv_color = False
        abs2meta_out = model._var_allprocs_abs2meta['output']
        var_sizes = model._var_sizes
        var_offsets = model._get_var_offsets()
        abs2idx = model._var_allprocs_abs2idx
        idx_iter_dict = {}  # a dict of index iterators

        simul_coloring = self.simul_coloring
        if simul_coloring:
            simul_color_modes = {'fwd': simul_coloring._fwd, 'rev': simul_coloring._rev}

        vois = self.input_meta[mode]
        input_list = self.input_list[mode]

        seed = []
        fwd = mode == 'fwd'

        loc_idxs = []
        idx_map = []
        start = 0
        end = 0

        # If we call compute_totals with any 'wrt' or 'of' that is outside of an existing driver
        # var set, then we need to ignore the computed relevancy and perform GS iterations on all
        # comps. Note, the inputs are handled individually by direct check vs the relevancy dict,
        # so we just bulk check the outputs here.
        qoi_i = self.input_meta[mode]
        qoi_o = self.output_meta[mode]
        if qoi_i and qoi_o:
            for out in self.output_list[mode]:
                if out not in qoi_o and out not in qoi_i:
                    non_rel_outs = True
                    break
            else:
                non_rel_outs = False
        else:
            non_rel_outs = False

        for name in input_list:
            if name in self.responses and self.responses[name]['alias'] is not None:
                path = self.responses[name]['source']
            else:
                path = name

            if path not in abs2meta_out:
                # could be promoted input name
                abs_in = model._var_allprocs_prom2abs_list['input'][path][0]
                path = model._conn_global_abs_in2out[abs_in]

            in_var_meta = abs2meta_out[path]

            if name in vois:
                # if name is in vois, then it has been declared as either a design var or
                # a constraint or an objective.
                meta = vois[name]
                if meta['distributed']:
                    end += meta['global_size']
                else:
                    end += meta['size']

                parallel_deriv_color = meta['parallel_deriv_color']
                cache_lin_sol = meta['cache_linear_solution']

                _check_voi_meta(name, parallel_deriv_color, simul_coloring)
                if parallel_deriv_color:
                    if parallel_deriv_color not in self.par_deriv_printnames:
                        self.par_deriv_printnames[parallel_deriv_color] = []

                    print_name = name
                    if name in self.ivc_print_names:
                        print_name = self.ivc_print_names[name]

                    self.par_deriv_printnames[parallel_deriv_color].append(print_name)

                in_idxs = meta['indices'] if 'indices' in meta else None

                if in_idxs is None:
                    # if the var is not distributed, global_size == local size
                    irange = np.arange(in_var_meta['global_size'], dtype=INT_DTYPE)
                else:
                    irange = in_idxs.shaped_array(copy=True)

            else:  # name is not a design var or response  (should only happen during testing)
                end += in_var_meta['global_size']
                irange = np.arange(in_var_meta['global_size'], dtype=INT_DTYPE)
                in_idxs = parallel_deriv_color = None
                cache_lin_sol = False

            in_var_idx = abs2idx[path]
            sizes = var_sizes['output']
            offsets = var_offsets['output']
            gstart = np.sum(sizes[:iproc, in_var_idx])
            gend = gstart + sizes[iproc, in_var_idx]

            # if we're doing parallel deriv coloring, we only want to set the seed on one proc
            # for each var in a given color
            if parallel_deriv_color:
                if fwd:
                    relev = relevant[name]['@all'][0]['output']
                else:
                    relev = relevant[name]['@all'][0]['input']
            else:
                relev = None

            dist = in_var_meta['distributed']
            if dist:
                if self.jac_dist_col_mask is None:
                    ndups = 1  # we don't divide by ndups for distributed inputs
                else:
                    ndups = np.nonzero(sizes[:, in_var_idx])[0].size
            else:
                # if the var is not distributed, convert the indices to global.
                # We don't iterate over the full distributed size in this case.
                irange += gstart

                if fwd or parallel_deriv_color:
                    ndups = 1
                else:
                    # find the number of duplicate components in rev mode so we can divide
                    # the seed between 'ndups' procs so that at the end after we do an
                    # Allreduce, the contributions from all procs will add up properly.
                    ndups = np.nonzero(sizes[:, in_var_idx])[0].size

            # all local idxs that correspond to vars from other procs will be -1
            # so each entry of loc_i will either contain a valid local index,
            # indicating we should set the local vector entry to 1.0 before running
            # solve_linear, or it will contain -1, indicating we should not set any
            # value before calling solve_linear.
            loc_i = np.full(irange.shape, -1, dtype=INT_DTYPE)
            if gend > gstart and (relev is None or relev):
                loc = np.nonzero(np.logical_and(irange >= gstart, irange < gend))[0]
                if in_idxs is None:
                    if in_var_meta['distributed']:
                        loc_i[loc] = np.arange(0, gend - gstart, dtype=INT_DTYPE)
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

            if parallel_deriv_color:
                has_par_deriv_color = True
                if parallel_deriv_color not in idx_iter_dict:
                    it = self.par_deriv_iter
                    imeta = defaultdict(bool)
                    imeta['par_deriv_color'] = parallel_deriv_color
                    imeta['idx_list'] = [(start, end)]
                    idx_iter_dict[parallel_deriv_color] = (imeta, it)
                else:
                    imeta, _ = idx_iter_dict[parallel_deriv_color]
                    imeta['idx_list'].append((start, end))
            elif not simul_coloring:  # plain old single index iteration
                imeta = defaultdict(bool)
                imeta['idx_list'] = range(start, end)
                idx_iter_dict[name] = (imeta, self.single_index_iter)

            if path in relevant and not non_rel_outs:
                relsystems = relevant[path]['@all'][1]
                if self.total_relevant_systems is not _contains_all:
                    self.total_relevant_systems.update(relsystems)
                tup = (ndups, relsystems, cache_lin_sol, name)
            else:
                self.total_relevant_systems = _contains_all
                tup = (ndups, _contains_all, cache_lin_sol, name)

            idx_map.extend([tup] * (end - start))
            start = end

        if has_par_deriv_color:
            _fix_pdc_lengths(idx_iter_dict)

        loc_idxs = np.hstack(loc_idxs)
        seed = np.hstack(seed)

        if simul_coloring and simul_color_modes[mode] is not None:
            imeta = defaultdict(bool)
            imeta['coloring'] = simul_coloring
            all_rel_systems = set()
            cache = False
            imeta['itermeta'] = itermeta = []
            locs = None
            for ilist in simul_coloring.color_iter(mode):
                for i in ilist:
                    _, rel_systems, cache_lin_sol, _ = idx_map[i]
                    _update_rel_systems(all_rel_systems, rel_systems)
                    cache |= cache_lin_sol

                iterdict = defaultdict(bool)

                if len(ilist) > 1:
                    locs = loc_idxs[ilist]
                    active = locs != -1
                    iterdict['local_in_idxs'] = locs[active]
                    iterdict['seeds'] = seed[ilist][active]

                iterdict['relevant'] = all_rel_systems
                iterdict['cache_lin_solve'] = cache
                itermeta.append(iterdict)

            idx_iter_dict['@simul_coloring'] = (imeta, self.simul_coloring_iter)

        self.in_idx_map[mode] = idx_map
        self.in_loc_idxs[mode] = loc_idxs
        self.idx_iter_dict[mode] = idx_iter_dict
        self.seeds[mode] = seed

    def _get_sol2jac_map(self, names, vois, allprocs_abs2meta_out, mode):
        """
        Create a dict mapping vecname and direction to an index array into the solution vector.

        Using the index array to pull values from the solution vector will give the values
        in the order needed by the jacobian.

        Parameters
        ----------
        names : iter of str
            Names of the variables making up the rows or columns of the jacobian.
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

        for name in names:

            drv_name = None
            if name in self.responses:
                path = self.responses[name]['source']
                drv_name = _src_or_alias_name(self.responses[name])
            else:
                path = name

            if name in vois:
                indices = vois[name]['indices']
                drv_name = _src_or_alias_name(vois[name])
            else:
                indices = None

            if drv_name is None:
                drv_name = name

            meta = allprocs_abs2meta_out[path]

            if indices is not None:
                sz = indices.indexed_src_size
            else:
                if self.get_remote:
                    sz = meta['global_size']
                else:
                    sz = meta['size']

            if (path in abs2idx and path in slices and path not in self.remote_vois):
                var_idx = abs2idx[path]
                slc = slices[path]

                if MPI and meta['distributed'] and self.get_remote:
                    if indices is not None:
                        local_idx, sizes_idx, _ = self._dist_driver_vars[drv_name]

                        dist_offset = np.sum(sizes_idx[:myproc])
                        full_inds = np.arange(slc.start, slc.stop, dtype=INT_DTYPE)
                        inds.append(full_inds[local_idx.as_array()])
                        jac_inds.append(jstart + dist_offset +
                                        np.arange(local_idx.indexed_src_size, dtype=INT_DTYPE))
                        name2jinds.append((path, jac_inds[-1]))
                    else:
                        dist_offset = np.sum(sizes[:myproc, var_idx])
                        inds.append(np.arange(slc.start, slc.stop, dtype=INT_DTYPE))
                        jac_inds.append(np.arange(jstart + dist_offset,
                                        jstart + dist_offset + sizes[myproc, var_idx],
                                        dtype=INT_DTYPE))
                        name2jinds.append((path, jac_inds[-1]))
                else:
                    idx_array = np.arange(slc.start, slc.stop, dtype=INT_DTYPE)
                    if indices is not None:
                        idx_array = idx_array[indices.flat()]
                    inds.append(idx_array)
                    jac_inds.append(np.arange(jstart, jstart + sz, dtype=INT_DTYPE))
                    if fwd or not self.get_remote:
                        name2jinds.append((path, jac_inds[-1]))

            if path not in self.remote_vois:
                jend += sz
                jstart = jend

        if inds:
            sol_idxs = np.hstack(inds)
            jac_idxs = np.hstack(jac_inds)
        else:
            sol_idxs = np.zeros(0, dtype=INT_DTYPE)
            jac_idxs = np.zeros(0, dtype=INT_DTYPE)

        return sol_idxs, jac_idxs, name2jinds

    def _get_tuple_map(self, names, vois, abs2meta_out):
        """
        Create a dict that maps var name to metadata tuple.

        The tuple has the form (jacobian row/column slice, indices, distrib)

        Parameters
        ----------
        names : iter of str
            Names of the variables making up the rows or columns of the jacobian.
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
        idx_map = {}
        start = 0
        end = 0
        get_remote = self.get_remote
        has_dist = False

        responses = vois if vois is self.responses else None

        for name in names:
            path = name

            if name in self.remote_vois:
                continue
            if name in vois:
                voi = vois[name]
                # this 'size'/'global_size' already takes indices into account
                if get_remote and voi['distributed']:
                    size = voi['global_size']
                else:
                    size = voi['size']
                indices = vois[name]['indices']
                if responses:
                    path = vois[name]['source']

            else:
                size = abs2meta_out[path]['global_size']
                indices = None

            has_dist |= abs2meta_out[path]['distributed']

            end += size

            idx_map[name] = (slice(start, end), indices, abs2meta_out[path]['distributed'])
            start = end

        return idx_map, end, has_dist  # after the loop, end is the total size

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
        """
        for i in imeta['idx_list']:
            yield i, self.single_input_setter, self.single_jac_setter, None

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
        """
        coloring = imeta['coloring']
        input_setter = self.simul_coloring_input_setter
        jac_setter = self.simul_coloring_jac_setter

        for color, ilist in enumerate(coloring.color_iter(mode)):
            if len(ilist) == 1:
                yield ilist, input_setter, jac_setter, None
            else:
                # yield all indices for a color at once
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
        _, rel_systems, cache_lin_sol, _ = self.in_idx_map[mode][idx]

        self._zero_vecs(mode)

        loc_idx = self.in_loc_idxs[mode][idx]
        if loc_idx >= 0:
            self.input_vec[mode].set_val(self.seeds[mode][idx], loc_idx)

        if cache_lin_sol:
            return rel_systems, ('linear',), (idx, mode)
        else:
            return rel_systems, None, None

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
        if itermeta is None:
            return self.single_input_setter(inds[0], None, mode)

        self._zero_vecs(mode)

        self.input_vec[mode].set_val(itermeta['seeds'], itermeta['local_in_idxs'])

        if itermeta['cache_lin_solve']:
            return itermeta['relevant'], ('linear',), (inds[0], mode)
        else:
            return itermeta['relevant'], None, None

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
        all_rel_systems = set()
        vec_names = set()

        dist = self.comm.size > 1

        for i in inds:
            if not dist or self.in_loc_idxs[mode][i] >= 0:
                rel_systems, vnames, _ = self.single_input_setter(i, imeta, mode)
                _update_rel_systems(all_rel_systems, rel_systems)
                if vnames is not None:
                    vec_names.add(vnames[0])

        if vec_names:
            return all_rel_systems, sorted(vec_names), (inds[0], mode)
        else:
            return all_rel_systems, None, None

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
        Scatter the i'th row or allreduce the i'th column of the jacobian.

        Parameters
        ----------
        i : int
            Total jacobian row or column index.
        mode : str
            Direction of derivative solution.
        """
        if self.get_remote and mode == 'fwd':
            if self.jac_scatters[mode] is not None:
                self.src_petsc[mode].array = self.J[:, i]
                self.tgt_petsc[mode].array[:] = self.J[:, i]
                self.jac_scatters[mode].scatter(self.src_petsc[mode], self.tgt_petsc[mode],
                                                addv=False, mode=False)
                self.J[:, i] = self.tgt_petsc[mode].array

        elif mode == 'rev':
            # for rows corresponding to serial 'of' vars, we need to correct for
            # duplication of their seed values by dividing by the number of duplications.
            ndups, _, _, oname = self.in_idx_map[mode][i]
            if self.get_remote:
                # don't use scratch[0] here because it may be in use by simul_coloring_jac_setter
                # which calls this method.
                scratch = self.jac_scratch['rev'][1]
                scratch[:] = self.J[i]

                self.comm.Allreduce(scratch, self.J[i], op=MPI.SUM)

                if ndups > 1:
                    if self.jac_dist_col_mask is not None:
                        scratch[:] = 1.0
                        scratch[self.jac_dist_col_mask] = (1.0 / ndups)
                        self.J[i] *= scratch
                    else:
                        self.J[i] *= (1.0 / ndups)
            else:
                scatter = self.jac_scatters[mode]
                if scatter is not None:
                    if self.dist_idx_map[mode][i]:  # distrib var, skip scatter
                        return
                    loc = self.loc_jac_idxs[mode][i]
                    if loc >= 0:
                        self.tgt_petsc[mode].array[:] = self.J[loc, :][self.nondist_loc_map[mode]]
                        self.src_petsc[mode].array[:] = self.J[loc, :][self.nondist_loc_map[mode]]
                    else:
                        self.src_petsc[mode].array[:] = 0.0

                    scatter.scatter(self.src_petsc[mode], self.tgt_petsc[mode],
                                    addv=True, mode=False)
                    if loc >= 0:
                        if ndups > 1:
                            self.J[loc, :][self.nondist_loc_map[mode]] = \
                                self.tgt_petsc[mode].array * (1.0 / ndups)
                        else:
                            self.J[loc, :][self.nondist_loc_map[mode]] = self.tgt_petsc[mode].array

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
        dist = self.comm.size > 1
        if dist:
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
            reduced_derivs = self.jac_scratch[mode][0]
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

    def compute_totals(self):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
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

        try:
            ln_solver = model._linear_solver
            with model._scaled_context_all():
                if len(model._subsystems_allprocs) > 0:
                    model._linearize(model._assembled_jac,
                                     sub_do_ln=ln_solver._linearize_children(),
                                     rel_systems=self.total_relevant_systems)
                else:
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
            for key, idx_info in self.idx_iter_dict[mode].items():
                imeta, idx_iter = idx_info
                for inds, input_setter, jac_setter, itermeta in idx_iter(imeta, mode):
                    rel_systems, vec_names, cache_key = input_setter(inds, itermeta, mode)

                    if debug_print:
                        if par_print and key in par_print:
                            varlist = '(' + ', '.join([name for name in par_print[key]]) + ')'
                            print('Solving color:', key, varlist, flush=True)
                        else:
                            if key == '@simul_coloring':
                                print(f'In mode: {mode}, Solving variable(s) using simul '
                                      'coloring:')
                                for local_ind in imeta['coloring']._local_indices(inds=inds,
                                                                                  mode=self.mode):
                                    print(f"   {local_ind}", flush=True)
                            else:
                                print(f"In mode: {mode}.\n('{self.ivc_print_names.get(key, key)}'"
                                      f", [{inds}])", flush=True)

                        t0 = time.perf_counter()

                    # restore old linear solution if cache_linear_solution was set by the user for
                    # any input variables involved in this linear solution.
                    with model._scaled_context_all():
                        if cache_key is not None and not has_lin_cons and self.mode == mode:
                            self._restore_linear_solution(cache_key, self.mode)
                            model._solve_linear(self.mode, rel_systems)
                            self._save_linear_solution(cache_key, self.mode)
                        else:
                            model._solve_linear(mode, rel_systems)

                    if debug_print:
                        print(f'Elapsed Time: {time.perf_counter() - t0} secs\n', flush=True)

                    jac_setter(inds, mode, imeta)

        # Driver scaling.
        if self.has_scaling:
            self._do_driver_scaling(self.J_dict)

        # if some of the wrt vars are distributed in fwd mode, we have to bcast from the rank
        # where each part of the distrib var exists
        if self.get_remote and mode == 'fwd' and self.has_input_dist[mode]:
            for start, stop, rank in self.dist_input_range_map[mode]:
                contig = self.J[:, start:stop].copy()
                model.comm.Bcast(contig, root=rank)
                self.J[:, start:stop] = contig

        if debug_print:
            # Debug outputs scaled derivatives.
            self._print_derivatives()

        return self.J_final

    def compute_totals_approx(self, initialize=False, progress_out_stream=None):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Uses an approximation method, e.g., fd or cs to calculate the derivatives.

        Parameters
        ----------
        initialize : bool
            Set to True to re-initialize the FD in model. This is only needed when manually
            calling compute_totals on the problem.
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

        model._tot_jac = self
        try:
            # Re-initialize so that it is clean.
            if initialize:

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
                model._setup_approx_partials()
                if model._coloring_info['coloring'] is not None:
                    model._update_wrt_matches(model._coloring_info)

            # Linearize Model
            if len(model._subsystems_allprocs) > 0:
                model._linearize(model._assembled_jac,
                                 sub_do_ln=model._linear_solver._linearize_children(),
                                 rel_systems=self.total_relevant_systems)
            else:
                model._linearize(model._assembled_jac,
                                 sub_do_ln=model._linear_solver._linearize_children())

        finally:
            model._tot_jac = None

        totals = self.J_dict
        if debug_print:
            print(f'Elapsed time to approx totals: {time.perf_counter() - t0} secs\n', flush=True)

        # Driver scaling.
        if self.has_scaling:
            self._do_driver_scaling(totals)

        if return_format == 'array':
            totals = self.J  # change back to array version

        if debug_print:
            # Debug outputs scaled derivatives.
            self._print_derivatives()

        return totals

    def _get_zero_inds(self, name, tup, jac_arr):
        """
        Get zero indices relative to the named variable for jac row/col 'jac_arr'.

        Parameters
        ----------
        name : str
            Name of the design var or response.
        tup : tuple
            Contains jacobian slice and dv/response indices, if any.
        jac_arr : ndarray
            Row or column of jacobian being checked for zero entries.

        Returns
        -------
        ndarray
            Index array of zero entries.
        """
        inds = tup[1]  # these must be indices into the flattened var
        shname = 'global_shape' if self.get_remote else 'shape'
        shape = self.model._var_allprocs_abs2meta['output'][name][shname]
        vslice = jac_arr[tup[0]]

        if inds is None:
            zero_idxs = np.nonzero(vslice.reshape(shape))
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
        col[nzrows] = False  # False in this case means nonzero
        if np.any(col):  # there's at least 1 row that's zero across all columns
            zero_rows = []
            for name, tup in self.of_meta.items():

                if name in self.responses:
                    name = self.responses[name]['source']

                zero_idxs = self._get_zero_inds(name, tup, col)

                if zero_idxs[0].size > 0:
                    if len(zero_idxs) == 1:
                        zero_rows.append((self.ivc_print_names.get(name, name),
                                          list(zero_idxs[0])))
                    else:
                        zero_rows.append((self.ivc_print_names.get(name, name),
                                          list(zip(*zero_idxs))))

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
            for name, tup in self.wrt_meta.items():

                if name in self.responses:
                    name = self.responses[name]['source']

                zero_idxs = self._get_zero_inds(name, tup, row)

                if zero_idxs[0].size > 0:
                    if len(zero_idxs) == 1:
                        zero_cols.append((self.ivc_print_names.get(name, name),
                                          list(zero_idxs[0])))
                    else:
                        zero_cols.append((self.ivc_print_names.get(name, name),
                                          list(zip(*zero_idxs))))

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
        desvars = self.prom_design_vars
        responses = self.prom_responses

        if self.return_format in ('dict', 'array'):
            for prom_out, odict in J.items():
                oscaler = responses[prom_out]['total_scaler']

                for prom_in, val in odict.items():
                    iscaler = desvars[prom_in]['total_scaler']

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
            raise RuntimeError("Derivative scaling by the driver only supports the 'dict' and "
                               "'array' formats at present.")

    def _print_derivatives(self):
        """
        Print out the derivatives when debug_print is True.
        """
        if self.return_format == 'dict':
            J_dict = self.J_dict
            for of, wrt_dict in J_dict.items():
                for wrt, J_sub in wrt_dict.items():
                    if wrt in self.ivc_print_names:
                        wrt = self.ivc_print_names[wrt]
                    pprint.pprint({(of, wrt): J_sub})
        else:
            J = self.J
            for i, of in enumerate(self.of):
                if of in self.remote_vois:
                    continue
                out_slice = self.of_meta[of][0]
                for j, wrt in enumerate(self.wrt):
                    if wrt not in self.remote_vois:
                        deriv = J[out_slice, self.wrt_meta[wrt][0]]
                        if wrt in self.ivc_print_names:
                            wrt = self.ivc_print_names[wrt]
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
            totals = self._get_dict_J(self.J, self.wrt, self.prom_wrt, self.of, self.prom_of,
                                      self.wrt_meta, self.of_meta, 'flat_dict_structured_key')
            requester._rec_mgr.record_derivatives(requester, totals, metadata)

        finally:
            self.model._recording_iter.pop()

    def set_col(self, icol, column):
        """
        Set the given column of the total jacobian.

        Parameters
        ----------
        icol : int
            Index of the column.
        column : ndarray
            Array to be copied into the jacobian column.
        """
        self.J[:, icol] = column


def _check_voi_meta(name, parallel_deriv_color, simul_coloring):
    """
    Check the contents of the given metadata for incompatible options.

    An exception will be raised if options are incompatible.

    Parameters
    ----------
    name : str
        Name of the variable.
    parallel_deriv_color : str
        Color of parallel deriv grouping.
    simul_coloring : ndarray
        Array of colors. Each entry corresponds to a column or row of the total jacobian.
    """
    if simul_coloring and parallel_deriv_color:
        raise RuntimeError("Using both simul_coloring and parallel_deriv_color with "
                           f"variable '{name}' is not supported.")


def _fix_pdc_lengths(idx_iter_dict):
    """
    Take any parallel_deriv_color entries and make sure their index arrays are same length.

    Parameters
    ----------
    idx_iter_dict : dict
        Dict of a name/color mapped to indexing information.
    """
    for imeta, _ in idx_iter_dict.values():
        par_deriv_color = imeta['par_deriv_color']
        range_list = imeta['idx_list']
        if par_deriv_color:
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


def _update_rel_systems(all_rel_systems, rel_systems):
    """
    Combine all relevant systems in those cases where we have multiple input variables involved.

    Parameters
    ----------
    all_rel_systems : set
        Current set of all relevant system names.
    rel_systems : set
        Set of relevant system names for the latest iteration.
    """
    if all_rel_systems is _contains_all or rel_systems is _contains_all:
        all_rel_systems = _contains_all
    else:
        all_rel_systems.update(rel_systems)
