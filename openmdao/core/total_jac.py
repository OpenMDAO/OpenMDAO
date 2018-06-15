"""
Helper class for total jacobian computation.
"""
from __future__ import print_function, division

import warnings
from collections import OrderedDict
from copy import deepcopy
import pprint
from six import iteritems
from six.moves import zip
import sys
import time

import numpy as np

from openmdao.utils.general_utils import ContainsAll
from openmdao.recorders.recording_iteration_stack import recording_iteration


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
    idx2local : dict
        Map of row/col index to index within a variable.  Used only if simul coloring is active.
    idx2name : dict
        Map of row/col index to variable name.  Used only if simul coloring is active.
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
    relevant : dict
        Map of absolute var name to vars that are relevant to it.
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
        self.relevant = model._relevant
        self.mode = problem._mode
        self.owning_ranks = problem.model._owning_rank
        self.has_scaling = driver._has_scaling and driver_scaling
        self.return_format = return_format
        self.lin_sol_cache = {}
        self.design_vars = design_vars = driver._designvars
        self.responses = responses = driver._responses
        self.debug_print = debug_print
        self.par_deriv = {}

        # Convert of and wrt names from promoted to absolute
        if wrt is None:
            wrt = prom_wrt = list(design_vars)
        else:
            prom_wrt = wrt
            if not global_names:
                wrt = [prom2abs[name][0] for name in prom_wrt]

        if of is None:
            of = list(driver._objs)
            of.extend(driver._cons)
            prom_of = of
        else:
            prom_of = of
            if not global_names:
                of = [prom2abs[name][0] for name in prom_of]

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

        self.of_meta, self.of_size = self._get_tuple_map(of, responses, abs2meta)
        self.wrt_meta, self.wrt_size = self._get_tuple_map(wrt, design_vars, abs2meta)

        if approx:
            self._initialize_approx(self.output_meta[self.mode])
        else:
            constraints = driver._cons

            for name in of:
                if name in constraints and constraints[name]['linear']:
                    has_lin_cons = True
                    break
            else:
                has_lin_cons = False

            self.simul_coloring = None if has_lin_cons else driver._simul_coloring_info

            if self.simul_coloring is None:
                modes = [self.mode]
            else:
                modes = [m for m in ('fwd', 'rev') if m in self.simul_coloring]
                # if our coloring has a 'best_mode' that differs from the specied mode,
                # change it to avoid having a coloring mismatch.
                if len(modes) == 1 and modes[0] != self.mode:
                    raise RuntimeError("Mode in coloring, '%s', differs from specified mode, '%s'."
                                       % (modes[0], self.mode))

            self.in_idx_map = {}
            self.in_loc_idxs = {}
            self.idx_iter_dict = {}

            self.out_meta = {}
            for mode in modes:
                self.in_idx_map[mode], self.in_loc_idxs[mode], self.idx_iter_dict[mode] = \
                    self._create_in_idx_map(has_lin_cons, mode)

                if mode == 'fwd':
                    self.out_meta[mode], out_size = self.of_meta, self.of_size
                else:
                    self.out_meta[mode], out_size = self.wrt_meta, self.wrt_size

            if not has_lin_cons and self.simul_coloring is not None:
                self.idx2name = {}
                self.idx2local = {}
                for mode in modes:
                    self.idx2name[mode], self.idx2local[mode] = self._create_idx_maps(
                        self.output_list[mode], self.output_meta[mode], out_size)
            else:
                self.idx2name = self.idx2local = self.simul_coloring = None

            self.has_lin_cons = has_lin_cons

            # always allocate a 2D dense array and we can assign views to dict keys later if
            # return format is 'dict' or 'flat_dict'.
            self.J = J = np.zeros((self.of_size, self.wrt_size))

            # for dict type return formats, map var names to views of the Jacobian array.
            if return_format == 'array':
                self.J_final = J
                if self.has_scaling:
                    # for array return format, create a 'dict' view for scaling only, since
                    # our scaling data is by variable.
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

    def _initialize_approx(self, output_meta):
        """
        Set up internal data structures needed for computing approx totals.

        Parameters
        ----------
        output_meta : dict
            Mapping of output name to response metadata (fwd) or design var metadata (rev).
        """
        model = self.model
        abs2meta = model._var_allprocs_abs2meta

        if self.return_format == 'array':
            self.J = J = np.zeros((self.of_size, self.wrt_size))

            # for array return format, create a 'dict' view so we can map partial subjacs into
            # the proper locations (and also do by-variable scaling if needed).
            self.J_dict = self._get_dict_J(J, self.wrt, self.prom_wrt, self.of, self.prom_of,
                                           self.wrt_meta, self.of_meta, 'dict')

        of_set = frozenset(self.of)
        wrt_set = frozenset(self.wrt)

        # Initialization based on driver (or user) -requested "of" and "wrt".
        if not model._owns_approx_jac or model._owns_approx_of != of_set \
           or model._owns_approx_wrt != wrt_set:
            model._owns_approx_of = of_set
            model._owns_approx_wrt = wrt_set

            # Support for indices defined on driver vars.
            model._owns_approx_of_idx = {
                key: val['indices'] for key, val in iteritems(self.responses)
                if val['indices'] is not None
            }
            model._owns_approx_wrt_idx = {
                key: val['indices'] for key, val in iteritems(self.design_vars)
                if val['indices'] is not None
            }

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
        else:
            raise ValueError("'%s' is not a valid jacobian return format." % return_format)

        return J_dict

    def _create_in_idx_map(self, has_lin_constraints, mode):
        """
        Create a list that maps a global index to a name, col/row range, and other data.

        Parameters
        ----------
        has_lin_constraints : bool
            If True, there are linear constraints used to compute the total jacobian.
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
        relevant = self.relevant
        has_par_deriv_color = False
        abs2meta = self.model._var_allprocs_abs2meta
        var_sizes = self.model._var_sizes
        abs2idx = self.model._var_allprocs_abs2idx
        idx_iter_dict = OrderedDict()  # a dict of index iterators

        simul_coloring = self.simul_coloring

        vois = self.input_meta[mode]
        input_list = self.input_list[mode]

        idx_tups = [None] * len(input_list)
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
            gstart = np.sum(sizes[:iproc, in_var_idx])
            gend = gstart + sizes[iproc, in_var_idx]

            if not in_var_meta['distributed']:
                # if the var is not distributed, convert the indices to global.
                # We don't iterate over the full distributed size in this case.
                owner = owning_ranks[name]
                if owner == iproc:
                    irange += gstart
                else:
                    owner_start = np.sum(sizes[:owner, in_var_idx])
                    irange += owner_start

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

            loc_idxs.append(loc_i)

            if name in relevant:
                rel = relevant[name]['@all'][1]
            else:
                rel = _contains_all

            if parallel_deriv_color:
                has_par_deriv_color = True
                if parallel_deriv_color not in idx_iter_dict:
                    if matmat:
                        it = self.par_deriv_matmat_iter
                    else:
                        it = self.par_deriv_iter
                    idx_iter_dict[parallel_deriv_color] = (parallel_deriv_color, matmat,
                                                           [(start, end)], it)
                else:
                    _, old_matmat, range_list, _ = idx_iter_dict[parallel_deriv_color]
                    if old_matmat != matmat:
                        raise RuntimeError("Mixing of vectorized and non-vectorized derivs in "
                                           "the same parallel color group (%s) is not "
                                           "supported." % parallel_deriv_color)
                    range_list.append((start, end))
            elif matmat:
                if name not in idx_iter_dict:
                    idx_iter_dict[name] = (None, matmat,
                                           [np.arange(start, end, dtype=int)],
                                           self.matmat_iter)
                else:
                    raise RuntimeError("Variable name '%s' matches a parallel_deriv_color "
                                       "name." % name)
            elif not simul_coloring:  # plain old single index iteration
                idx_iter_dict[name] = (None, False,
                                       np.arange(start, end, dtype=int),
                                       self.single_index_iter)

            tup = (name, rhsname, rel, cache_lin_sol)
            idx_map.extend([tup] * (end - start))
            start = end

        if has_par_deriv_color:
            _fix_pdc_lengths(idx_iter_dict)

        if simul_coloring:
            idx_iter_dict['@simul_coloring'] = (False, False, self.simul_coloring,
                                                self.simul_coloring_iter)

        return idx_map, np.hstack(loc_idxs), idx_iter_dict

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

    def _create_idx_maps(self, names, vois, size):
        """
        Create a list that maps jacobian row/column index to var name.

        Parameters
        ----------
        names : iter of str
            Names of the variables making up the rows or columns of the jacobian.
        vois : dict
            Mapping of variable of interest (desvar or response) name to its metadata.
        size : int
            Total number of rows/columns.

        Returns
        -------
        list
            List that maps row/col index to variable name.
        ndarray
            Array that maps row/col index to local variable index
        """
        idx2name = [None] * size
        idx2local = np.empty(size, dtype=int)
        abs2meta = self.model._var_allprocs_abs2meta

        start = end = 0
        for name in names:
            if name in vois:
                end += vois[name]['size']
            else:
                end += abs2meta[name]['global_size']

            idx2name[start:end] = [name] * (end - start)
            idx2local[start:end] = np.arange(0, end - start, dtype=int)
            start = end

        return idx2name, idx2local

    #
    # outer loop iteration functions
    #
    def single_index_iter(self, idxs):
        """
        Iterate over single indices for a single variable.

        Parameters
        ----------
        idxs : iter of int
            Total jacobian row/column indices.

        Yields
        ------
        int
            Current index.
        method
            Input setter method.
        method
            Jac setter method.
        """
        for i in idxs:
            yield i, self.single_input_setter, self.single_jac_setter, self.mode

    def simul_coloring_iter(self, coloring_info):
        """
        Iterate over index lists for the simul coloring case.

        Parameters
        ----------
        coloring_info : tuple of the form (column_or_row_lists, row_or_column_map, sparsity)
            Row/column data needed to group colors and associate rows and columns.

        Yields
        ------
        list of int or int
            Current indices or current index.
        method
            Input setter method.
        method
            Jac setter method.
        """
        modes = [k for k in ('fwd', 'rev') if k in coloring_info]

        input_setter = self.simul_coloring_input_setter
        jac_setter = self.simul_coloring_jac_setter

        # do all the colored rows/cols
        for mode in modes:
            for color, ilist in enumerate(coloring_info[mode][0]):
                if color > 0:
                    # yield all indices for a color at once
                    yield ilist, input_setter, jac_setter, mode

        # do all of the uncolored rows/cols last so they can overwrite any wrong values
        for mode in modes:
            ilist = coloring_info[mode][0][0]
            for i in ilist:
                # do all non-colored indices individually (one linear solve per index)
                yield i, self.single_input_setter, self.single_jac_setter, mode

    def par_deriv_iter(self, idxs):
        """
        Iterate over index lists for the parallel deriv case.

        Parameters
        ----------
        idxs : iter of int
            Total jacobian row/column indices.

        Yields
        ------
        list of int
            Current indices.
        method
            Input setter method.
        method
            Jac setter method.
        """
        for tup in zip(*idxs):
            yield tup, self.par_deriv_input_setter, self.par_deriv_jac_setter, self.mode

    def matmat_iter(self, idxs):
        """
        Iterate over index lists for the matrix matrix case.

        Parameters
        ----------
        idxs : ndarray of int
            Total jacobian row/column indices.

        Yields
        ------
        list of int
            Current indices.
        method
            Input setter method.
        method
            Jac setter method.
        """
        for idx_list in idxs:
            yield idx_list, self.matmat_input_setter, self.matmat_jac_setter, self.mode

    def par_deriv_matmat_iter(self, idxs):
        """
        Iterate over index lists for the combined parallel deriv matrix matrix case.

        Parameters
        ----------
        idxs : iter of int
            Total jacobian row/column indices.

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
        yield idxs, self.par_deriv_matmat_input_setter, self.par_deriv_matmat_jac_setter, self.mode

    #
    # input setter functions
    #
    def single_input_setter(self, idx, mode):
        """
        Set 1's into the input vector in the single index case.

        Parameters
        ----------
        idx : int
            Total jacobian row or column index.
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
        input_name, vecname, rel_systems, cache_lin_sol = self.in_idx_map[mode][idx]

        loc_idx = self.in_loc_idxs[mode][idx]
        if loc_idx != -1:
            self.input_vec[mode][vecname]._views_flat[input_name][loc_idx] = -1.0

        if cache_lin_sol:
            return rel_systems, (vecname,), (idx, mode)
        else:
            return rel_systems, None, None

    def simul_coloring_input_setter(self, inds, mode):
        """
        Set 1's into the input vector in the simul coloring case.

        Parameters
        ----------
        inds : list of int
            Total jacobian row or column indices.
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
        all_rel_systems = set()
        cache = False

        for i in inds:
            rel_systems, vec_names, _ = self.single_input_setter(i, mode)
            _update_rel_systems(all_rel_systems, rel_systems)
            cache |= vec_names is not None

        if cache:
            return all_rel_systems, ('linear',), (inds[0], mode)
        else:
            return all_rel_systems, None, None

    def par_deriv_input_setter(self, inds, mode):
        """
        Set 1's into the input vector in the parallel derivative case.

        Parameters
        ----------
        inds : tuple of int
            Total jacobian row or column indices.
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
        vec_names = []

        for count, i in enumerate(inds):
            rel_systems, vnames, _ = self.single_input_setter(i, mode)
            _update_rel_systems(all_rel_systems, rel_systems)
            if vnames is not None:
                vec_names.append(vnames[0])

        if vec_names:
            return all_rel_systems, vec_names, (inds[0], mode)
        else:
            return all_rel_systems, None, None

    def matmat_input_setter(self, inds, mode):
        """
        Set 1's into the input vector in the matrix-matrix case.

        Parameters
        ----------
        inds : ndarray of int
            Total jacobian row or column indices.
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

        input_name, vec_name, rel_systems, cache_lin_sol = in_idx_map[inds[0]]

        dinputs = input_vec[vec_name]

        for col, i in enumerate(inds):
            loc_idx = in_loc_idxs[i]
            if loc_idx != -1:
                dinputs._views_flat[input_name][loc_idx, col] = -1.0

        if cache_lin_sol:
            return rel_systems, (vec_name,), (inds[0], mode)
        else:
            return rel_systems, None, None

    def par_deriv_matmat_input_setter(self, inds, mode):
        """
        Set 1's into the input vector in the matrix matrix with parallel deriv case.

        Parameters
        ----------
        inds : list of ndarray of int
            Total jacobian row or column indices.
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

        vec_names = []
        for matmat_idxs in inds:
            input_name, vec_name, rel_systems, cache_lin_sol = in_idx_map[matmat_idxs[0]]
            if cache_lin_sol:
                vec_names.append(vec_name)
            cache |= cache_lin_sol
            _update_rel_systems(all_rel_systems, rel_systems)

            dinputs = input_vec[vec_name]
            ncol = dinputs._ncol

            for col, i in enumerate(matmat_idxs):
                loc_idx = in_loc_idxs[i]
                if loc_idx != -1:
                    if ncol > 1:
                        dinputs._views_flat[input_name][loc_idx, col] = -1.0
                    else:
                        dinputs._views_flat[input_name][loc_idx] = -1.0

        if cache:
            return all_rel_systems, vec_names, (inds[0][0], mode)
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
        input_name, vecname, _, _ = self.in_idx_map[mode][i]
        out_views = self.output_vec[mode][vecname]._views_flat
        relevant = self.relevant
        fwd = mode == 'fwd'
        J = self.J
        nproc = self.comm.size
        out_meta = self.out_meta[mode]

        for output_name in self.output_list[mode]:
            if input_name not in relevant or output_name in relevant[input_name]:
                slc, indices, distrib = out_meta[output_name]
                deriv_val = None
                if output_name in out_views:
                    deriv_val = out_views[output_name]
                    if indices is not None:
                        deriv_val = deriv_val[indices]

                if nproc > 1 and not distrib:
                    if deriv_val is None:
                        deriv_val = np.empty(slc.stop - slc.start)
                    self.comm.Bcast(deriv_val, root=self.owning_ranks[output_name])

                if fwd:
                    J[slc, i] = deriv_val
                else:
                    J[i, slc] = deriv_val

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
        row_col_map = self.simul_coloring[mode][1]
        out_meta = self.out_meta[mode]
        idx2name = self.idx2name[mode]
        idx2local = self.idx2local[mode]
        outvecs = self.output_vec[mode]
        in_idx_map = self.in_idx_map[mode]
        fwd = mode == 'fwd'

        J = self.J

        for i in inds:
            input_name, vecname, _, _ = in_idx_map[i]
            out_views = outvecs[vecname]._views_flat
            for row_or_col in row_col_map[i]:
                output_name = idx2name[row_or_col]
                deriv_val = None
                if output_name in out_views:
                    deriv_val = out_views[output_name]
                    indices = out_meta[output_name][1]
                    if indices is not None:
                        deriv_val = deriv_val[indices]
                    # print("deriv_val:", i, output_name, input_name, deriv_val)
                    if fwd:
                        J[row_or_col, i] = deriv_val[idx2local[row_or_col]]
                    else:
                        J[i, row_or_col] = deriv_val[idx2local[row_or_col]]

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
        input_name, vecname, _, _ = self.in_idx_map[mode][inds[0]]
        out_views = self.output_vec[mode][vecname]._views_flat
        ncol = self.output_vec[mode][vecname]._ncol
        relevant = self.relevant
        nproc = self.comm.size
        fwd = self.mode == 'fwd'
        J = self.J
        out_meta = self.out_meta[mode]

        for output_name in self.output_list[mode]:
            slc, indices, distrib = out_meta[output_name]
            deriv_val = out_idxs = None
            if input_name not in relevant or output_name in relevant[input_name]:
                if output_name in out_views:
                    deriv_val = out_views[output_name]
                    if indices is not None:
                        deriv_val = deriv_val[indices]

                if nproc > 1 and not distrib:
                    if deriv_val is None:
                        deriv_val = np.empty((slc.stop - slc.start, ncol))
                    self.comm.Bcast(deriv_val, root=self.owning_ranks[output_name])

                if fwd:
                    J[slc, inds] = deriv_val
                else:
                    J[inds, slc] = deriv_val.T

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
        recording_iteration.stack.append(('_compute_totals', 0))
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
            vec_dinput[vec_name].set_const(0.0)
            vec_doutput[vec_name].set_const(0.0)
            vec_dresid[vec_name].set_const(0.0)

        # Linearize Model
        model._linearize(model._assembled_jac, sub_do_ln=model._linear_solver._linearize_children())
        model._linear_solver._linearize()

        # Main loop over columns (fwd) or rows (rev) of the jacobian
        for key, meta in iteritems(self.idx_iter_dict[self.mode]):
            _, _, idxs, idx_iter = meta
            for inds, input_setter, jac_setter, mode in idx_iter(idxs):
                # this sets dinputs for the current par_deriv_color to 0
                # dinputs is dresids in fwd, doutouts in rev
                vec_doutput['linear'].set_const(0.0)
                if mode == 'fwd':
                    vec_dresid['linear'].set_const(0.0)
                else:  # rev
                    vec_dinput['linear'].set_const(0.0)

                rel_systems, vec_names, cache_key = input_setter(inds, mode)

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

        recording_iteration.stack.pop()

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
        recording_iteration.stack.append(('_compute_totals', 0))

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

        # Linearize Model
        model._linearize(model._assembled_jac, sub_do_ln=model._linear_solver._linearize_children())
        model._linear_solver._linearize()

        approx_jac = model._jacobian._subjacs_info

        of_idx = model._owns_approx_of_idx
        wrt_idx = model._owns_approx_wrt_idx

        if return_format == 'flat_dict':
            totals = OrderedDict()
            for prom_out, output_name in zip(self.prom_of, of):
                for prom_in, input_name in zip(self.prom_wrt, wrt):
                    totals[prom_out, prom_in] = _get_subjac(approx_jac[output_name, input_name],
                                                            prom_out, prom_in, of_idx, wrt_idx)

        elif return_format == 'dict':
            totals = OrderedDict()
            for prom_out, output_name in zip(self.prom_of, of):
                totals[prom_out] = tot = OrderedDict()
                for prom_in, input_name in zip(self.prom_wrt, wrt):
                    tot[prom_in] = _get_subjac(approx_jac[output_name, input_name],
                                               prom_out, prom_in, of_idx, wrt_idx)

        elif return_format == 'array':
            totals = self.J_dict  # J_dict has views into the array jacobian
            for prom_out, output_name in zip(self.prom_of, of):
                tot = totals[prom_out]
                for prom_in, input_name in zip(self.prom_wrt, wrt):
                    tot[prom_in][:] = _get_subjac(approx_jac[output_name, input_name],
                                                  prom_out, prom_in, of_idx, wrt_idx)
        else:
            msg = "Unsupported return format '%s." % return_format
            raise NotImplementedError(msg)

        if self.has_scaling:
            self._do_scaling(totals)

        if return_format == 'array':
            totals = self.J  # change back to array version

        recording_iteration.stack.pop()
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
                for vs in doutputs._data:
                    doutputs._data[vs][:] = save_vec[vs]
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
            for vs, data in iteritems(doutputs._data):
                save_vec[vs][:] = data

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
    for key, tup in iteritems(idx_iter_dict):
        par_deriv_color, matmat, range_list, _ = tup
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
