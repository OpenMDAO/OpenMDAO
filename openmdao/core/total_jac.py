"""
Helper class for total jacobian computation.
"""
from __future__ import print_function, division

import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
import pprint
from six import iteritems, itervalues
from six.moves import zip
import sys
import time

import numpy as np

try:
    from petsc4py import PETSc
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETSc = None

from openmdao.vectors.vector import INT_DTYPE
from openmdao.utils.general_utils import ContainsAll, simple_warning
from openmdao.utils.record_util import create_local_meta
from openmdao.utils.mpi import MPI
from openmdao.approximation_schemes.approximation_scheme import _initialize_model_approx
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian


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
    out_meta : dict
        Map of absoute output var name to tuples of the form (row/column slice, indices, distrib).
    of_meta : dict
        Map of absoute output 'of' var name to tuples of the form
        (row/column slice, indices, distrib).
    wrt_meta : dict
        Map of absoute output 'wrt' var name to tuples of the form
        (row/column slice, indices, distrib).
    output_list : list of str
        List of names of output variables for this total jacobian.  In fwd mode, outputs
        are responses.  In rev mode, outputs are design variables.
    output_vec : Dict of vectors keyed by vec_name.
        Designated output vectors based on value of fwd.
    owning_ranks : dict
        Map of absolute var name to the MPI process that owns it.
    par_deriv : dict
        Cache containing names of desvars or responses for each parallel derivative color.
    return_format : str
        Indicates the desired return format of the total jacobian. Can have value of
        'array', 'dict', or 'flat_dict'.
    simul_coloring : tuple of the form (column_lists, row_map, sparsity) or None
        Contains all data necessary to simultaneously solve for groups of total derivatives.
    """

    def __init__(self, problem, of, wrt, global_names, return_format, approx=False,
                 debug_print=False, driver_scaling=True):
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
        global_names : bool
            If True, names in of and wrt are global names.
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
        """
        driver = problem.driver
        prom2abs = problem.model._var_allprocs_prom2abs_list['output']

        self.model = model = problem.model
        self.comm = problem.comm
        self.mode = problem._mode
        self.owning_ranks = problem.model._owning_rank
        self.has_scaling = driver._has_scaling and driver_scaling
        self.return_format = return_format
        self.lin_sol_cache = {}
        self.design_vars = design_vars = driver._designvars
        self.responses = responses = driver._responses
        self.debug_print = debug_print
        self.par_deriv = {}
        self._recording_iter = driver._recording_iter

        if not model._use_derivatives:
            raise RuntimeError("Derivative support has been turned off but compute_totals "
                               "was called.")

        driver_wrt = list(design_vars)
        driver_of = driver._get_ordered_nl_responses()

        # Convert of and wrt names from promoted to absolute
        if wrt is None:
            if driver_wrt:
                wrt = prom_wrt = driver_wrt
            else:
                raise RuntimeError("Driver is not providing any design variables "
                                   "for compute_totals.")
        else:
            prom_wrt = wrt
            if not global_names:
                wrt = [prom2abs[name][0] for name in prom_wrt]

        if of is None:
            if driver_of:
                of = prom_of = driver_of
            else:
                raise RuntimeError("Driver is not providing any response variables "
                                   "for compute_totals.")
        else:
            prom_of = of
            if not global_names:
                of = [prom2abs[name][0] for name in prom_of]

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
        self.input_vec = {'fwd': model._vectors['residual'], 'rev': model._vectors['output']}
        self.output_vec = {'fwd': model._vectors['output'], 'rev': model._vectors['residual']}

        abs2meta = model._var_allprocs_abs2meta

        if approx:
            _initialize_model_approx(model, driver, self.of, self.wrt)
        else:
            constraints = driver._cons

            for name in of:
                if name in constraints and constraints[name]['linear']:
                    has_lin_cons = True
                    self.simul_coloring = None
                    break
            else:
                has_lin_cons = False
                self.simul_coloring = driver._total_coloring

                # if we don't get wrt and of from driver, turn off coloring
                if self.simul_coloring is not None and (wrt != driver_wrt or of != driver_of):
                    msg = ("compute_totals called using a different list of design vars and/or "
                           "responses than those used to define coloring, so coloring will "
                           "be turned off.\ncoloring design vars: %s, current design vars: "
                           "%s\ncoloring responses: %s, current responses: %s." %
                           (driver_wrt, wrt, driver_of, of))
                    simple_warning(msg)
                    self.simul_coloring = None

            self.has_lin_cons = has_lin_cons

            if self.simul_coloring is None:
                modes = [self.mode]
            else:
                modes = self.simul_coloring.modes()

            self.in_idx_map = {}
            self.in_loc_idxs = {}
            self.idx_iter_dict = {}

            for mode in modes:
                self.in_idx_map[mode], self.in_loc_idxs[mode], self.idx_iter_dict[mode] = \
                    self._create_in_idx_map(mode)

        self.of_meta, self.of_size = self._get_tuple_map(of, responses, abs2meta)
        self.wrt_meta, self.wrt_size = self._get_tuple_map(wrt, design_vars, abs2meta)
        self.out_meta = {'fwd': self.of_meta, 'rev': self.wrt_meta}

        # always allocate a 2D dense array and we can assign views to dict keys later if
        # return format is 'dict' or 'flat_dict'.
        self.J = J = np.zeros((self.of_size, self.wrt_size))

        if not approx:
            self.solvec_map = {}
            for mode in modes:
                self.solvec_map[mode] = self._get_solvec_map(self.output_list[mode],
                                                             self.output_meta[mode],
                                                             abs2meta, mode)
            self.jac_scatters = {}
            self.jac_petsc = {}
            self.soln_petsc = {}
            if 'fwd' in modes:
                self._compute_jac_scatters('fwd', J.shape[0])

            if 'rev' in modes:
                self._compute_jac_scatters('rev', J.shape[1])

        # for dict type return formats, map var names to views of the Jacobian array.
        if return_format == 'array':
            self.J_final = J
            if self.has_scaling or approx:
                # for array return format, create a 'dict' view for scaling or FD, since
                # our scaling and FD data is by variable.
                self.J_dict = self._get_dict_J(J, wrt, prom_wrt, of, prom_of,
                                               self.wrt_meta, self.of_meta, 'dict')
            else:
                self.J_dict = None
        else:
            self.J_final = self.J_dict = self._get_dict_J(J, wrt, prom_wrt, of, prom_of,
                                                          self.wrt_meta, self.of_meta,
                                                          return_format)

        if self.has_scaling:
            self.prom_design_vars = {prom_wrt[i]: design_vars[dv] for i, dv in enumerate(wrt)}
            self.prom_responses = {prom_of[i]: responses[r] for i, r in enumerate(of)}

    def _compute_jac_scatters(self, mode, size):
        rank = self.comm.rank
        self.jac_scatters[mode] = jac_scatters = {}
        model = self.model

        if self.comm.size > 1 or (model._full_comm is not None and
                                  model._full_comm.size > 1):
            tgt_vec = PETSc.Vec().createWithArray(np.zeros(size, dtype=float),
                                                  comm=self.comm)
            self.jac_petsc[mode] = tgt_vec
            self.soln_petsc[mode] = {}
            sol_idxs, jac_idxs = self.solvec_map[mode]
            for vecname in model._lin_vec_names:
                src_arr = self.output_vec[mode][vecname]._data
                if isinstance(self.output_vec[mode][vecname], PETScVector):
                    src_vec = self.output_vec[mode][vecname]._petsc
                else:
                    outvec = self.output_vec[mode][vecname]
                    if outvec._ncol == 1:
                        src_vec = PETSc.Vec().createWithArray(src_arr, comm=self.comm)
                    else:
                        src_vec = PETSc.Vec().createWithArray(
                            self.output_vec[mode][vecname]._data[:, 0].copy(),
                            comm=self.comm)
                self.soln_petsc[mode][vecname] = (src_vec, src_arr)

                offset = size * rank
                jac_inds = jac_idxs[vecname]
                if isinstance(jac_idxs[vecname], slice):
                    jac_inds = np.arange(offset, offset + size, dtype=INT_DTYPE)
                else:
                    jac_inds += offset
                src_indexset = PETSc.IS().createGeneral(sol_idxs[vecname], comm=self.comm)
                tgt_indexset = PETSc.IS().createGeneral(jac_inds, comm=self.comm)
                jac_scatters[vecname] = PETSc.Scatter().create(src_vec, src_indexset,
                                                               tgt_vec, tgt_indexset)
        else:
            for vecname in model._lin_vec_names:
                jac_scatters[vecname] = None

    def _get_dict_J(self, J, wrt, prom_wrt, of, prom_of, wrt_meta, of_meta, return_format):
        """
        Create a dict or flat-dict jacobian that maps to views in the given 2D array jacobian.

        Parameters
        ----------
        J : ndarray
            Array jacobian.
        wrt : iter of str
            Absolute names of input vars.
        prom_wrt : iter of str
            Promoted names of input vars.
        of : iter of str
            Absolute names of output vars.
        prom_of : iter of str
            Promoted names of output vars.
        wrt_meta : dict
            Dict mapping input name to array jacobian slice, indices, and distrib.
        of_meta : dict
            Dict mapping output name to array jacobian slice, indices, and distrib.
        return_format : str
            Indicates the desired form of the returned jacobian.

        Returns
        -------
        OrderedDict
            Dict form of the total jacobian that contains views of the ndarray jacobian.
        """
        J_dict = OrderedDict()
        if return_format == 'dict':
            for i, out in enumerate(of):
                J_dict[prom_of[i]] = outer = OrderedDict()
                out_slice = of_meta[out][0]
                for j, inp in enumerate(wrt):
                    outer[prom_wrt[j]] = J[out_slice, wrt_meta[inp][0]]
        elif return_format == 'flat_dict':
            for i, out in enumerate(of):
                out_slice = of_meta[out][0]
                for j, inp in enumerate(wrt):
                    J_dict[prom_of[i], prom_wrt[j]] = J[out_slice, wrt_meta[inp][0]]
        elif return_format == 'flat_dict_structured_key':
            # This format is supported by the recorders (specifically the sql recorder), which use
            # numpy structured arrays.
            for i, out in enumerate(of):
                out_slice = of_meta[out][0]
                for j, inp in enumerate(wrt):
                    key = "%s,%s" % (prom_of[i], prom_wrt[j])
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

        Returns
        -------
        list
            List of (name, rhsname, rel_systems) tuples.
        ndarray
            array of local indices
        dict
            dictionary of iterators.
        """
        iproc = self.comm.rank
        owning_ranks = self.owning_ranks
        model = self.model
        relevant = model._relevant
        has_par_deriv_color = False
        abs2meta = model._var_allprocs_abs2meta
        var_sizes = model._var_sizes
        var_offsets = model._get_var_offsets()
        abs2idx = model._var_allprocs_abs2idx
        idx_iter_dict = OrderedDict()  # a dict of index iterators

        simul_coloring = self.simul_coloring
        if simul_coloring:
            simul_color_modes = {'fwd': simul_coloring._fwd, 'rev': simul_coloring._rev}

        vois = self.input_meta[mode]
        input_list = self.input_list[mode]

        loc_idxs = []
        idx_map = []
        start = 0
        end = 0

        for name in input_list:
            rhsname = 'linear'
            in_var_meta = abs2meta[name]

            if name in vois:
                # if name is in vois, then it has been declared as either a design var or
                # a constraint or an objective.
                meta = vois[name]
                end += meta['size']

                parallel_deriv_color = meta['parallel_deriv_color']
                matmat = meta['vectorize_derivs']
                cache_lin_sol = meta['cache_linear_solution']

                _check_voi_meta(name, parallel_deriv_color, matmat, simul_coloring)
                if matmat or parallel_deriv_color:
                    rhsname = name

                    if parallel_deriv_color and self.debug_print:
                        if parallel_deriv_color not in self.par_deriv:
                            self.par_deriv[parallel_deriv_color] = []
                        self.par_deriv[parallel_deriv_color].append(name)

                in_idxs = meta['indices'] if 'indices' in meta else None

                if in_idxs is None:
                    # if the var is not distributed, global_size == local size
                    irange = np.arange(in_var_meta['global_size'], dtype=int)
                else:
                    irange = in_idxs
                    # correct for any negative indices
                    irange[in_idxs < 0] += in_var_meta['global_size']

            else:  # name is not a design var or response  (should only happen during testing)
                end += in_var_meta['global_size']
                irange = np.arange(in_var_meta['global_size'], dtype=int)
                in_idxs = parallel_deriv_color = matmat = None
                cache_lin_sol = False

            in_var_idx = abs2idx[rhsname][name]
            sizes = var_sizes[rhsname]['output']
            offsets = var_offsets[rhsname]['output']
            gstart = np.sum(sizes[:iproc, in_var_idx])
            gend = gstart + sizes[iproc, in_var_idx]

            if not in_var_meta['distributed']:
                # if the var is not distributed, convert the indices to global.
                # We don't iterate over the full distributed size in this case.
                owner = owning_ranks[name]
                if owner == iproc:
                    irange += gstart
                else:
                    irange += np.sum(sizes[:owner, in_var_idx])

            # all local idxs that correspond to vars from other procs will be -1
            # so each entry of loc_i will either contain a valid local index,
            # indicating we should set the local vector entry to 1.0 before running
            # solve_linear, or it will contain -1, indicating we should not set any
            # value before calling solve_linear.
            loc_i = np.full(irange.shape, -1, dtype=int)
            if gend > gstart:
                loc = np.nonzero(np.logical_and(irange >= gstart, irange < gend))[0]
                if in_idxs is None:
                    if in_var_meta['distributed']:
                        loc_i[loc] = np.arange(0, gend - gstart, dtype=int)
                    else:
                        loc_i[loc] = irange[loc] - gstart
                else:
                    loc_i[loc] = irange[loc]
                    if not in_var_meta['distributed']:
                        loc_i[loc] -= gstart

                loc_offset = offsets[iproc, in_var_idx] - offsets[iproc, 0]
                loc_i[loc] += loc_offset

            loc_idxs.append(loc_i)

            if parallel_deriv_color:
                has_par_deriv_color = True
                if parallel_deriv_color not in idx_iter_dict:
                    if matmat:
                        it = self.par_deriv_matmat_iter
                    else:
                        it = self.par_deriv_iter
                    imeta = defaultdict(bool)
                    imeta['par_deriv_color'] = parallel_deriv_color
                    imeta['matmat'] = matmat
                    imeta['idx_list'] = [(start, end)]
                    idx_iter_dict[parallel_deriv_color] = (imeta, it)
                else:
                    imeta, _ = idx_iter_dict[parallel_deriv_color]
                    if imeta['matmat'] != matmat:
                        raise RuntimeError('Mixing of vectorized and non-vectorized derivs in '
                                           'the same parallel color group (%s) is not '
                                           'supported.' % parallel_deriv_color)
                    imeta['idx_list'].append((start, end))
            elif matmat:
                if name not in idx_iter_dict:
                    imeta = defaultdict(bool)
                    imeta['matmat'] = matmat
                    imeta['idx_list'] = [np.arange(start, end, dtype=int)]
                    idx_iter_dict[name] = (imeta, self.matmat_iter)
                else:
                    raise RuntimeError("Variable name '%s' matches a parallel_deriv_color "
                                       "name." % name)
            elif not simul_coloring:  # plain old single index iteration
                imeta = defaultdict(bool)
                imeta['idx_list'] = np.arange(start, end, dtype=int)
                idx_iter_dict[name] = (imeta, self.single_index_iter)

            if name in relevant:
                tup = (rhsname, relevant[name]['@all'][1], cache_lin_sol)
            else:
                tup = (rhsname, _contains_all, cache_lin_sol)

            idx_map.extend([tup] * (end - start))
            start = end

        if has_par_deriv_color:
            _fix_pdc_lengths(idx_iter_dict)

        loc_idxs = np.hstack(loc_idxs)

        if simul_coloring and simul_color_modes[mode] is not None:
            imeta = defaultdict(bool)
            imeta['coloring'] = simul_coloring
            all_rel_systems = set()
            cache = False
            imeta['itermeta'] = itermeta = []
            locs = None
            for ilist in simul_coloring.color_iter(mode):
                for i in ilist:
                    _, rel_systems, cache_lin_sol = idx_map[i]
                    _update_rel_systems(all_rel_systems, rel_systems)
                    cache |= cache_lin_sol

                iterdict = defaultdict(bool)

                if len(ilist) > 1:
                    locs = loc_idxs[ilist]
                    iterdict['local_in_idxs'] = locs[locs != -1.0]

                iterdict['relevant'] = all_rel_systems
                iterdict['cache_lin_solve'] = cache
                itermeta.append(iterdict)

            idx_iter_dict['@simul_coloring'] = (imeta, self.simul_coloring_iter)

        return idx_map, loc_idxs, idx_iter_dict

    def _get_solvec_map(self, names, vois, abs2meta, mode):
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
        abs2meta : dict
            Mapping of absolute var name to metadata for that var.
        mode : str
            Derivative solution direction.

        Returns
        -------
        dict
            Mapping of vecname to index array for all names in order
        """
        idxs = {}
        jac_idxs = {}
        model = self.model
        myproc = model.comm.rank
        owners = model._owning_rank
        fwd = mode == 'fwd'
        missing = False
        full_slice = slice(None)

        for vecname in model._lin_vec_names:
            inds = []
            jac_inds = []
            sizes = model._var_sizes[vecname]['output']
            offsets = model._var_offsets[vecname]['output']
            abs2idx = model._var_allprocs_abs2idx[vecname]
            start = end = 0

            for name in names:
                indices = vois[name]['indices'] if name in vois else None
                meta = abs2meta[name]

                if name in abs2idx:
                    var_idx = abs2idx[name]
                    if meta['distributed']:
                        # if var is distributed, we need all of its parts from all procs
                        dist_idxs = []
                        for rank in range(model.comm.size):
                            if sizes[rank, var_idx] > 0:
                                offset = offsets[rank, var_idx]
                                dist_idxs.append(np.arange(offset, offset + sizes[rank, var_idx],
                                                           dtype=INT_DTYPE))
                        idx_array = np.hstack(dist_idxs)
                    else:
                        iproc = owners[name]

                        offset = offsets[iproc, var_idx]
                        idx_array = np.arange(offset, offset + sizes[iproc, var_idx],
                                              dtype=INT_DTYPE)
                        if indices is not None:
                            idx_array = idx_array[indices]

                    sz = idx_array.size
                else:
                    missing = True
                    sz = meta['global_size']

                if indices is not None:
                    sz = len(indices)

                end += sz
                if name in abs2idx:
                    inds.append(idx_array)
                    jac_inds.append((start, end))

                start = end

            idxs[vecname] = np.hstack(inds)

            if missing:
                jac_idxs[vecname] = np.hstack([np.arange(start, end, dtype=INT_DTYPE)
                                               for start, end in jac_inds])
            else:
                jac_idxs[vecname] = full_slice

        return idxs, jac_idxs

    def _get_tuple_map(self, names, vois, abs2meta):
        """
        Create a dict that maps var name to metadata tuple.

        The tuple has the form (jacobian row/column slice, indices, distrib)

        Parameters
        ----------
        names : iter of str
            Names of the variables making up the rows or columns of the jacobian.
        vois : dict
            Mapping of variable of interest (desvar or response) name to its metadata.
        abs2meta : dict
            Mapping of absolute var name to metadata for that var.

        Returns
        -------
        dict
            Dict of metadata tuples keyed by output name.
        int
            Total number of rows or columns.
        """
        idx_map = {}
        start = 0
        end = 0

        for name in names:
            if name in vois:
                # this 'size' already takes indices into account
                size = vois[name]['size']
                indices = vois[name]['indices']
            else:
                size = abs2meta[name]['global_size']
                indices = None

            end += size

            idx_map[name] = (slice(start, end), indices, abs2meta[name]['distributed'])
            start = end

        return idx_map, end  # after the loop, end is the total size

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
        coloring_info = imeta['coloring']
        both = coloring_info._fwd and coloring_info._rev
        input_setter = self.simul_coloring_input_setter
        jac_setter = self.simul_coloring_jac_setter

        for color, ilist in enumerate(coloring_info.color_iter(mode)):
            if len(ilist) == 1:
                if both:
                    yield ilist, input_setter, jac_setter, None
                else:
                    yield ilist[0], self.single_input_setter, self.single_jac_setter, None
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
            yield tup, self.par_deriv_input_setter, self.par_deriv_jac_setter, None

    def matmat_iter(self, imeta, mode):
        """
        Iterate over index lists for the matrix matrix case.

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
        for idx_list in imeta['idx_list']:
            yield idx_list, self.matmat_input_setter, self.matmat_jac_setter, None

    def par_deriv_matmat_iter(self, imeta, mode):
        """
        Iterate over index lists for the combined parallel deriv matrix matrix case.

        Parameters
        ----------
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Yields
        ------
        list of ndarray of int
            Current indices.
        method
            Input setter method.
        method
            Jac setter method.
        """
        # here, idxs is a list of arrays.  One array in the list for each parallel deriv
        # variable, and the entries in each array are all of the indices corresponding
        # to that variable's rows or columns in the total jacobian.
        idxs = imeta['idx_list']
        yield idxs, self.par_deriv_matmat_input_setter, self.par_deriv_matmat_jac_setter, None

    def _zero_vecs(self, vecname, mode):
        vecs = self.model._vectors

        # clean out vectors from last solve
        vecs['output'][vecname]._data[:] = 0.0
        vecs['residual'][vecname]._data[:] = 0.0
        if mode == 'rev':
            vecs['input'][vecname]._data[:] = 0.0

    #
    # input setter functions
    #
    def single_input_setter(self, idx, imeta, mode):
        """
        Set -1's into the input vector in the single index case.

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
        tuple or None
            vec_name corresponding to the given index (or None).
        int or None
            key used for storage of cached linear solve (if active, else None).
        """
        vecname, rel_systems, cache_lin_sol = self.in_idx_map[mode][idx]

        self._zero_vecs(vecname, mode)

        loc_idx = self.in_loc_idxs[mode][idx]
        if loc_idx >= 0:
            # We apply a -1 here because the derivative of the output is minus the derivative of
            # the residual in openmdao.
            self.input_vec[mode][vecname]._data[loc_idx] = -1.0

        if cache_lin_sol:
            return rel_systems, (vecname,), (idx, mode)
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

        self._zero_vecs('linear', mode)

        # We apply a -1 here because the derivative of the output is minus the derivative of
        # the residual in openmdao.
        self.input_vec[mode]['linear']._data[itermeta['local_in_idxs']] = -1.0

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

        for i in inds:
            rel_systems, vnames, _ = self.single_input_setter(i, imeta, mode)
            _update_rel_systems(all_rel_systems, rel_systems)
            if vnames is not None:
                vec_names.add(vnames[0])

        if vec_names:
            return all_rel_systems, sorted(vec_names), (inds[0], mode)
        else:
            return all_rel_systems, None, None

    def matmat_input_setter(self, inds, imeta, mode):
        """
        Set -1's into the input vector in the matrix-matrix case.

        Parameters
        ----------
        inds : ndarray of int
            Total jacobian row or column indices.
        imeta : dict
            Dictionary of iteration metadata.
        mode : str
            Direction of derivative solution.

        Returns
        -------
        set
            Set of relevant system names.
        tuple of str or None
            (vec_name,) or None if linear solve caching is inactive.
        int or None
            key used for storage of cached linear solve (if active, else None).
        """
        input_vec = self.input_vec[mode]
        in_idx_map = self.in_idx_map[mode]
        in_loc_idxs = self.in_loc_idxs[mode]

        vec_name, rel_systems, cache_lin_sol = in_idx_map[inds[0]]

        self._zero_vecs(vec_name, mode)

        dinputs = input_vec[vec_name]

        for col, i in enumerate(inds):
            loc_idx = in_loc_idxs[i]
            if loc_idx != -1:
                # We apply a -1 here because the derivative of the output is minus the derivative
                # of the residual in openmdao.
                dinputs._data[loc_idx, col] = -1.0

        if cache_lin_sol:
            return rel_systems, (vec_name,), (inds[0], mode)
        else:
            return rel_systems, None, None

    def par_deriv_matmat_input_setter(self, inds, imeta, mode):
        """
        Set -1's into the input vector in the matrix matrix with parallel deriv case.

        Parameters
        ----------
        inds : list of ndarray of int
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
            vec_names or None if linear solve caching is inactive.
        int or None
            key used for storage of cached linear solve (if active, else None).
        """
        input_vec = self.input_vec[mode]
        in_idx_map = self.in_idx_map[mode]
        in_loc_idxs = self.in_loc_idxs[mode]

        all_rel_systems = set()
        cache = False

        vec_names = set()
        for matmat_idxs in inds:
            vec_name, rel_systems, cache_lin_sol = in_idx_map[matmat_idxs[0]]
            if cache_lin_sol:
                vec_names.add(vec_name)
            cache |= cache_lin_sol
            _update_rel_systems(all_rel_systems, rel_systems)

            self._zero_vecs(vec_name, mode)

            dinputs = input_vec[vec_name]
            ncol = dinputs._ncol

            for col, i in enumerate(matmat_idxs):
                loc_idx = in_loc_idxs[i]
                if loc_idx != -1:

                    # We apply a -1 here because the derivative of the output is minus the
                    # derivative of the residual in openmdao.
                    if ncol > 1:
                        dinputs._data[loc_idx, col] = -1.0
                    else:
                        dinputs._data[loc_idx] = -1.0

        if cache:
            return all_rel_systems, sorted(vec_names), (inds[0][0], mode)
        else:
            return all_rel_systems, None, None

    #
    # Jacobian setter functions
    #
    def single_jac_setter(self, i, mode):
        """
        Set the appropriate part of the total jacobian for a single input index.

        Parameters
        ----------
        i : int
            Total jacobian row or column index.
        mode : str
            Direction of derivative solution.
        """
        vecname, _, _ = self.in_idx_map[mode][i]

        scatter = self.jac_scatters[mode][vecname]

        # DefaultVector
        if scatter is None:
            deriv_idxs, jac_idxs = self.solvec_map[mode]
            deriv_val = self.output_vec[mode][vecname]._data
            if mode == 'fwd':
                self.J[jac_idxs[vecname], i] = deriv_val[deriv_idxs[vecname]]
            else:  # rev
                self.J[i, jac_idxs[vecname]] = deriv_val[deriv_idxs[vecname]]

        # PETScVector
        else:
            self.jac_petsc[mode].array[:] = 0.
            scatter.scatter(self.soln_petsc[mode][vecname][0],
                            self.jac_petsc[mode], addv=False, mode=False)
            if mode == 'fwd':
                self.J[:, i] = self.jac_petsc[mode].array
            else:
                self.J[i] = self.jac_petsc[mode].array

    def par_deriv_jac_setter(self, inds, mode):
        """
        Set the appropriate part of the total jacobian for multiple input indices.

        Parameters
        ----------
        inds : tuple of int
            Total jacobian row or column indices.
        mode : str
            Direction of derivative solution.
        """
        for i in inds:
            self.single_jac_setter(i, mode)

    def simul_coloring_jac_setter(self, inds, mode):
        """
        Set the appropriate part of the total jacobian for simul coloring input indices.

        Parameters
        ----------
        inds : list of int
            Total jacobian row or column indices.
        mode : str
            Direction of derivative solution.
        """
        row_col_map = self.simul_coloring.get_row_col_map(mode)
        fwd = mode == 'fwd'

        J = self.J
        deriv_idxs, _ = self.solvec_map[mode]

        # because simul_coloring cannot be used with vectorized derivs (matmat) or parallel
        # deriv coloring, vecname will always be 'linear', and we don't need to check
        # vecname for each index.
        scatter = self.jac_scatters[mode]['linear']

        if scatter is None:
            deriv_val = self.output_vec[mode]['linear']._data
            reduced_derivs = deriv_val[deriv_idxs['linear']]
        else:
            self.jac_petsc[mode].array[:] = 0.
            scatter.scatter(self.soln_petsc[mode]['linear'][0],
                            self.jac_petsc[mode], addv=False, mode=False)
            reduced_derivs = self.jac_petsc[mode].array

        if fwd:
            for i in inds:
                J[row_col_map[i], i] = reduced_derivs[row_col_map[i]]
        else:
            for i in inds:
                J[i, row_col_map[i]] = reduced_derivs[row_col_map[i]]

    def matmat_jac_setter(self, inds, mode):
        """
        Set the appropriate part of the total jacobian for matrix matrix input indices.

        Parameters
        ----------
        inds : ndarray of int
            Total jacobian row or column indices.
        mode : str
            Direction of derivative solution. (ignored)
        """
        # in plain matmat, all inds are for a single variable for each iteration of the outer loop,
        # so any relevance can be determined only once.
        vecname, _, _ = self.in_idx_map[mode][inds[0]]
        ncol = self.output_vec[mode][vecname]._ncol
        nproc = self.comm.size
        fwd = self.mode == 'fwd'
        J = self.J
        out_meta = self.out_meta[mode]

        deriv_val = self.output_vec[mode][vecname]._data
        deriv_idxs, jac_idxs = self.solvec_map[mode]
        scatter = self.jac_scatters[mode][vecname]
        jac_inds = jac_idxs[vecname]
        if scatter is None:
            deriv_val = deriv_val[deriv_idxs[vecname], :]
            if mode == 'fwd':
                for col, i in enumerate(inds):
                    self.J[jac_inds, i] = deriv_val[:, col]
            else:  # rev
                for col, i in enumerate(inds):
                    self.J[i, jac_inds] = deriv_val[:, col]
        else:
            solution = self.soln_petsc[mode][vecname]
            for col, i in enumerate(inds):
                self.jac_petsc[mode].array[:] = 0.
                if ncol > 1:
                    solution[0].array = solution[1][:, col]
                else:
                    solution[0].array = solution[1]
                scatter.scatter(self.soln_petsc[mode][vecname][0],
                                self.jac_petsc[mode], addv=False, mode=False)
                if mode == 'fwd':
                    self.J[:, i] = self.jac_petsc[mode].array
                else:
                    self.J[i] = self.jac_petsc[mode].array

    def par_deriv_matmat_jac_setter(self, inds, mode):
        """
        Set the appropriate part of the total jacobian for par_deriv matrix matrix input indices.

        Parameters
        ----------
        inds : list of ndarray of int
            Total jacobian row or column indices.
        mode : str
            Direction of derivative solution. (ignored)
        """
        for matmat_idxs in inds:
            self.matmat_jac_setter(matmat_idxs, mode)

    def compute_totals(self):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        debug_print = self.debug_print
        par_deriv = self.par_deriv

        has_lin_cons = self.has_lin_cons

        model = self.model
        vec_dinput = model._vectors['input']
        vec_doutput = model._vectors['output']
        vec_dresid = model._vectors['residual']

        # Prepare model for calculation by cleaning out the derivatives
        # vectors.
        for vec_name in model._lin_vec_names:
            vec_dinput[vec_name]._data[:] = 0.0
            vec_doutput[vec_name]._data[:] = 0.0
            vec_dresid[vec_name]._data[:] = 0.0

        # Linearize Model
        with model._scaled_context_all():
            model._linearize(model._assembled_jac,
                             sub_do_ln=model._linear_solver._linearize_children())
        model._linear_solver._linearize()

        # Main loop over columns (fwd) or rows (rev) of the jacobian
        for mode in self.idx_iter_dict:
            for key, idx_info in iteritems(self.idx_iter_dict[mode]):
                imeta, idx_iter = idx_info
                for inds, input_setter, jac_setter, itermeta in idx_iter(imeta, mode):

                    rel_systems, vec_names, cache_key = input_setter(inds, itermeta, mode)

                    if debug_print:
                        if par_deriv and key in par_deriv:
                            varlist = '(' + ', '.join([name for name in par_deriv[key]]) + ')'
                            print('Solving color:', key, varlist)
                        else:
                            print('Solving variable:', key)

                        sys.stdout.flush()
                        t0 = time.time()

                    # restore old linear solution if cache_linear_solution was set by the user for
                    # any input variables involved in this linear solution.
                    with model._scaled_context_all():
                        if cache_key is not None and not has_lin_cons:
                            self._restore_linear_solution(vec_names, cache_key, self.mode)
                            model._solve_linear(model._lin_vec_names, self.mode, rel_systems)
                            self._save_linear_solution(vec_names, cache_key, self.mode)
                        else:
                            model._solve_linear(model._lin_vec_names, mode, rel_systems)

                    if debug_print:
                        print('Elapsed Time:', time.time() - t0, '\n')
                        sys.stdout.flush()

                    jac_setter(inds, mode)

        if self.has_scaling:
            self._do_scaling(self.J_dict)

        if debug_print:
            # Debug outputs scaled derivatives.
            self._print_derivatives()

        # np.save("total_jac.npy", self.J)

        return self.J_final

    def compute_totals_approx(self, initialize=False):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Uses an approximation method, e.g., fd or cs to calculate the derivatives.

        Parameters
        ----------
        initialize : bool
            Set to True to re-initialize the FD in model. This is only needed when manually
            calling compute_totals on the problem.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        of = self.of
        wrt = self.wrt
        model = self.model
        return_format = self.return_format

        # Prepare model for calculation by cleaning out the derivatives
        # vectors.
        for vec_name in model._lin_vec_names:
            model._vectors['input'][vec_name].set_const(0.0)
            model._vectors['output'][vec_name].set_const(0.0)
            model._vectors['residual'][vec_name].set_const(0.0)

        # Solve for derivs with the approximation_scheme.
        # This cuts out the middleman by grabbing the Jacobian directly after linearization.

        # Re-initialize so that it is clean.
        if initialize:
            if model._approx_schemes:
                method = list(model._approx_schemes)[0]
                kwargs = model._owns_approx_jac_meta
                model.approx_totals(method=method, **kwargs)
            else:
                model.approx_totals(method='fd')

            model._setup_jacobians(recurse=False)
            if model._approx_schemes or model._approx_coloring_info:
                model._setup_approx_partials()

        # Linearize Model
        model._linearize(model._assembled_jac,
                         sub_do_ln=model._linear_solver._linearize_children())

        approx_jac = model._jacobian._subjacs_info

        of_idx = model._owns_approx_of_idx
        wrt_idx = model._owns_approx_wrt_idx

        totals = self.J_dict
        if return_format == 'flat_dict':
            for prom_out, output_name in zip(self.prom_of, of):
                for prom_in, input_name in zip(self.prom_wrt, wrt):
                    totals[prom_out, prom_in][:] = _get_subjac(approx_jac[output_name, input_name],
                                                               prom_out, prom_in, of_idx, wrt_idx)

        elif return_format in ('dict', 'array'):
            for prom_out, output_name in zip(self.prom_of, of):
                tot = totals[prom_out]
                for prom_in, input_name in zip(self.prom_wrt, wrt):
                    if prom_out == prom_in and isinstance(tot[prom_in], dict):
                        rows, cols, data = tot[prom_in]['coo']
                        data[:] = _get_subjac(approx_jac[output_name, input_name],
                                              prom_out, prom_in, of_idx, wrt_idx)[rows, cols]
                    else:
                        tot[prom_in][:] = _get_subjac(approx_jac[output_name, input_name],
                                                      prom_out, prom_in, of_idx, wrt_idx)
        else:
            msg = "Unsupported return format '%s." % return_format
            raise NotImplementedError(msg)

        if self.has_scaling:
            self._do_scaling(totals)

        if return_format == 'array':
            totals = self.J  # change back to array version

        return totals

    def _restore_linear_solution(self, vec_names, key, mode):
        """
        Restore the previous linear solution.

        Parameters
        ----------
        vec_names : list of str
            Names of output vectors to restore.
        key : hashable object
            Key to lookup linear solution.
        mode : str
            Direction of derivative solution.
        """
        lin_sol_cache = self.lin_sol_cache
        if key in lin_sol_cache:
            lin_sol = lin_sol_cache[key]
            for i, vec_name in enumerate(vec_names):
                save_vec = lin_sol[i]
                doutputs = self.output_vec[mode][vec_name]
                doutputs._data[:] = save_vec
        else:
            lin_sol_cache[key] = lin_sol = []
            for vec_name in vec_names:
                lin_sol.append(deepcopy(self.output_vec[mode][vec_name]._data))

    def _save_linear_solution(self, vec_names, key, mode):
        """
        Save the current linear solution.

        Parameters
        ----------
        vec_names : list of str
            Names of output vectors to restore.
        key : hashable object
            Key to lookup linear solution.
        mode : str
            Direction of derivative solution.
        """
        lin_sol = self.lin_sol_cache[key]
        for i, vec_name in enumerate(vec_names):
            save_vec = lin_sol[i]
            doutputs = self.output_vec[mode][vec_name]
            save_vec[:] = doutputs._data

    def _do_scaling(self, J):
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
            for prom_out, odict in iteritems(J):
                oscaler = responses[prom_out]['scaler']

                for prom_in, val in iteritems(odict):
                    iscaler = desvars[prom_in]['scaler']

                    # Scale response side
                    if oscaler is not None:
                        val[:] = (oscaler * val.T).T

                    # Scale design var side
                    if iscaler is not None:
                        val *= 1.0 / iscaler

        elif self.return_format == 'flat_dict':
            for tup, val in iteritems(J):
                prom_out, prom_in = tup
                oscaler = responses[prom_out]['scaler']
                iscaler = desvars[prom_in]['scaler']

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
            J = self.J_dict
            for of in self.of:
                for wrt in self.wrt:
                    pprint.pprint({(of, wrt): J[of][wrt]})
        else:
            J = self.J
            for i, of in enumerate(self.of):
                out_slice = self.of_meta[of][0]
                for j, wrt in enumerate(self.wrt):
                    pprint.pprint({(of, wrt): J[out_slice, self.wrt_meta[wrt][0]]})

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
        self._recording_iter.stack.append((requester._get_name(), requester.iter_count))

        try:
            totals = self._get_dict_J(self.J, self.wrt, self.prom_wrt, self.of, self.prom_of,
                                      self.wrt_meta, self.of_meta, 'flat_dict_structured_key')
            requester._rec_mgr.record_derivatives(requester, totals, metadata)

        finally:
            self._recording_iter.stack.pop()


def _get_subjac(jac_meta, prom_out, prom_in, of_idx, wrt_idx):
    """
    Return proper subjacobian based on input/output names and indices.

    Parameters
    ----------
    jac_meta : dict
        Partial subjacobian metadata coming from approx_jac.
    prom_out : str
        Promoted output name.
    prom_in : str
        Promoted input name.
    of_idx : dict
        Mapping of promoted output name to indices.
    wrt_idx : dict
        Mapping of promoted input name to indices.

    Returns
    -------
    ndarray
        The desired subjacobian.
    """
    if jac_meta['rows'] is not None:  # sparse list format
        # This is a design variable that was declared as an obj/con.
        tot = np.eye(len(jac_meta['value']))
        if prom_out in of_idx:
            tot = tot[of_idx[prom_out], :]
        if prom_in in wrt_idx:
            tot = tot[:, wrt_idx[prom_in]]
        return tot
    else:
        return jac_meta['value']


def _check_voi_meta(name, parallel_deriv_color, matmat, simul_coloring):
    """
    Check the contents of the given metadata for incompatible options.

    An exception will be raised if options are incompatible.

    Parameters
    ----------
    name : str
        Name of the variable.
    parallel_deriv_color : str
        Color of parallel deriv grouping.
    matmat : bool
        If True, vectorize derivatives for this variable.
    simul_coloring : ndarray
        Array of colors. Each entry corresponds to a column or row of the total jacobian.
    """
    if simul_coloring:
        if parallel_deriv_color:
            raise RuntimeError("Using both simul_coloring and parallel_deriv_color with "
                               "variable '%s' is not supported." % name)
        if matmat:
            raise RuntimeError("Using both simul_coloring and vectorize_derivs with "
                               "variable '%s' is not supported." % name)


def _fix_pdc_lengths(idx_iter_dict):
    """
    Take any parallel_deriv_color entries and make sure their index arrays are same length.

    Parameters
    ----------
    idx_iter_dict : dict
        Dict of a name/color mapped to indexing information.
    """
    for imeta, _ in itervalues(idx_iter_dict):
        par_deriv_color = imeta['par_deriv_color']
        matmat = imeta['matmat']
        range_list = imeta['idx_list']
        if par_deriv_color:
            if not matmat:
                lens = np.array([end - start for start, end in range_list])
                maxlen = np.max(lens)
                diffs = lens - maxlen
            if not matmat and np.any(diffs):
                for i, diff in enumerate(diffs):
                    start, end = range_list[i]
                    if diff < 0:
                        range_list[i] = np.empty(maxlen, dtype=int)
                        range_list[i][:end - start] = np.arange(start, end, dtype=int)
                        range_list[i][end - start:] = range_list[i][end - start - 1]
                    else:
                        range_list[i] = np.arange(start, end, dtype=int)
            else:
                # just convert all (start, end) tuples to aranges
                for i, (start, end) in enumerate(range_list):
                    range_list[i] = np.arange(start, end, dtype=int)


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
