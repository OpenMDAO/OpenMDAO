from __future__ import print_function, division

from collections import OrderedDict
import numpy as np
from six import iteritems, itervalues

from openmdao.utils.general_utils import ContainsAll


_contains_all = ContainsAll()


class _TotalJacInfo(object):
    def __init__(self, problem, of, wrt, global_names, return_format):
        model = problem.model
        self.comm = problem.comm
        self.nproc = problem.comm.size
        self.iproc = model.comm.rank
        self.var_sizes = model._var_sizes
        self.relevant = model._relevant
        self.fwd = (problem._mode == 'fwd')
        self.abs2idx = model._var_allprocs_abs2idx
        self.abs2meta = model._var_allprocs_abs2meta
        self.owning_ranks = problem.model._owning_rank
        self.return_format = return_format

        design_vars = problem.driver._designvars
        responses = problem.driver._responses
        constraints = problem.driver._cons

        vec_doutput = model._vectors['output']
        vec_dresid = model._vectors['residual']

        prom2abs = model._var_allprocs_prom2abs_list['output']

        # Convert of and wrt names from promoted to unpromoted
        # (which is absolute path since we're at the top)
        if wrt is None:
            wrt = old_wrt = list(design_vars)
        else:
            old_wrt = wrt
            if not global_names:
                wrt = [prom2abs[name][0] for name in old_wrt]

        if of is None:
            of = list(problem.driver._objs)
            of.extend(problem.driver._cons)
            old_of = of
        else:
            old_of = of
            if not global_names:
                of = [prom2abs[name][0] for name in old_of]

        self.of = of
        self.old_of = old_of
        self.wrt = wrt
        self.old_wrt = old_wrt

        if self.fwd:
            self.input_list, self.output_list = wrt, of
            self.input_meta, self.output_meta = design_vars, responses
            self.input_vec, self.output_vec = vec_dresid, vec_doutput
        else:  # rev
            self.input_list, self.output_list = of, wrt
            self.input_meta, self.output_meta = responses, design_vars
            self.input_vec, self.output_vec = vec_doutput, vec_dresid

        for name in of:
            if name in constraints and constraints[name]['linear']:
                has_lin_cons = True
                break
        else:
            has_lin_cons = False

        self.output_slice_map, self.out_size = self._create_slice_map(self.output_list,
                                                                      self.output_meta)
        self.simul_coloring = None
        if not has_lin_cons and problem.driver._simul_coloring_info is not None:
            self.simul_coloring = problem.driver._simul_coloring_info
            self.idx2name, self.idx2local = self._create_idx_maps(self.output_list, self.output_meta,
                                                                  self.out_size)
        else:
            self.idx2name = self.idx2local = None

        self.input_idx_map, self.input_loc_idxs, self.idx_iter_dict = \
            self._create_input_idx_map(self.input_list, self.input_meta, has_lin_cons)

        # always allocate a 2D dense array and we can assign views to dict keys later if
        # return format is 'dict' or 'flat_dict'.
        if self.fwd:
            self.J = J = np.zeros((self.out_size, self.input_loc_idxs.size))
        else:  # rev
            self.J = J = np.zeros((self.input_loc_idxs.size, self.out_size))

        # for dict type return formats, map var names to views of the Jacobian array.
        if return_format == 'array':
            self.Jfinal = J
        else:
            self.Jfinal = self._get_dict_J(J, wrt, old_wrt, of, old_of,
                                           self.input_meta, self.output_slice_map, return_format)


    def _get_dict_J(self, J, wrt, oldwrt, of, oldof, input_meta, output_slice_map, return_format):
        """
        Create a dict or flat-dict jacobian that maps to views in the given 2D array jacobian.

        Parameters
        ----------
        J : ndarray
            Array jacobian.
        wrt : iter of str
            Absolute names of input vars.
        oldwrt : iter of str
            Promoted names of input vars.
        of : iter of str
            Absolute names of output vars.
        oldof : iter of str
            Promoted names of output vars.
        input_meta : dict
            Dict of input voi metadata.
        output_slice_map : dict
            Dict mapping output name to array jacobian slice.
        return_format : str
            Indicates the desired form of the returned jacobian.
        """
        if self.fwd:
            input_slice_map, in_size = self._create_slice_map(wrt, input_meta)
            slice_map_in, slice_map_out = input_slice_map, output_slice_map
        else:
            input_slice_map, in_size = self._create_slice_map(of, input_meta)
            slice_map_in, slice_map_out = output_slice_map, input_slice_map

        Jdict = OrderedDict()
        if return_format == 'dict':
            for i, out in enumerate(of):
                Jdict[oldof[i]] = outer = OrderedDict()
                out_slice = slice_map_out[out]
                for j, inp in enumerate(wrt):
                    outer[oldwrt[j]] = J[out_slice, slice_map_in[inp]]
        elif return_format == 'flat_dict':
            for i, out in enumerate(of):
                out_slice = slice_map_out[out]
                for j, inp in enumerate(wrt):
                    Jdict[oldof[i], oldwrt[j]] = J[out_slice, slice_map_in[inp]]
        else:
            raise ValueError("'%s' is not a valid jacobian return format." % return_format)

        return Jdict


    def _create_input_idx_map(self, names, vois, has_lin_constraints):
        """
        Create a list that maps a global index to a name, col/row range, and other data.

        Parameters
        ----------
        names : iter of str
            Names of the variables making up the rows or columns of the jacobian.
        vois : dict
            Mapping of variable of interest (desvar or response) name to its metadata.

        Returns
        -------
        list
            List of (name, rhsname, rel_systems) tuples.
        ndarray
            array of local indices
        dict
            dictionary of iterators.
        """
        idx_map = []
        start = 0
        end = 0
        var_sizes = self.var_sizes
        abs2idx = self.abs2idx
        abs2meta = self.abs2meta
        iproc = self.iproc
        owning_ranks = self.owning_ranks
        relevant = self.relevant
        has_par_deriv_color = False

        idx_tups = [None] * len(names)
        loc_idxs = []
        idx_iter_dict = {}  # a dict of index iterators

        simul_coloring = self.simul_coloring

        for i, name in enumerate(names):
            rhsname = 'linear'
            in_var_meta = abs2meta[name]

            if name in vois:
                meta = vois[name]
                end += meta['size']

                parallel_deriv_color = meta['parallel_deriv_color']
                matmat = meta['vectorize_derivs']
                _check_voi_meta(name, parallel_deriv_color, matmat, simul_coloring)
                if matmat or parallel_deriv_color:
                    rhsname = name

                in_idxs = meta['indices'] if 'indices' in meta else None

                if in_idxs is None:
                    # if the var is not distributed, global_size == local size
                    irange = np.arange(in_var_meta['global_size'], dtype=int)
                else:
                    irange = in_idxs
                    # correct for any negative indices
                    irange[in_idxs < 0] += in_var_meta['global_size']

            else:  # name is not a VOI  (should only happen during testing)
                end += abs2meta[name]['size']
                irange = np.arange(in_var_meta['global_size'], dtype=int)
                in_idxs = parallel_deriv_color = matmat = None

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
            loc = np.logical_and(irange >= gstart, irange < gend)
            if in_idxs is None:
                loc_i[loc] = np.arange(0, gend - gstart, dtype=int)[loc]
            else:
                loc_i[loc] = irange[loc]

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
                        raise RuntimeError("Mixing of vectorized and non-vectorized derivs in the "
                                           "same parallel color group (%s) is not supported." %
                                           parallel_deriv_color)
                    range_list.append((start, end))
            elif matmat:
                if name not in idx_iter_dict:
                    idx_iter_dict[name] = (parallel_deriv_color, matmat,
                                           [np.arange(start, end, dtype=int)], self.matmat_iter)
                else:
                    raise RuntimeError("Variable name '%s' matches a parallel_deriv_color name." %
                                       name)
            elif not simul_coloring:
                idx_iter_dict[name] = (parallel_deriv_color, matmat,
                                       np.arange(start, end, dtype=int), self.single_index_iter)

            tup = (name, rhsname, rel)
            idx_map.extend([tup] * (end - start))
            start = end

        if has_par_deriv_color:
            _fix_pdc_lengths(idx_iter_dict)

        if simul_coloring:
            idx_iter_dict['@simul_coloring'] = (False, False, self.simul_coloring,
                                                self.simul_coloring_iter)

        return idx_map, np.hstack(loc_idxs), idx_iter_dict

    def _create_slice_map(self, names, vois):
        """
        Create a dict that maps var name to a jacobian row or column slice.

        Parameters
        ----------
        names : iter of str
            Names of the variables making up the rows or columns of the jacobian.
        vois : dict
            Mapping of variable of interest (desvar or response) name to its metadata.

        Returns
        -------
        dict
            Dict of slice keyed by output name.
        int
            Total number of rows or columns.
        """
        idx_map = {}
        start = 0
        end = 0
        abs2meta = self.abs2meta

        for name in names:
            if name in vois:
                end += vois[name]['size']
            else:
                end += abs2meta[name]['size']

            idx_map[name] = slice(start, end)
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
        abs2meta = self.abs2meta

        start = end = 0
        for name in names:
            if name in vois:
                end += vois[name]['size']
            else:
                end += abs2meta[name]['size']

            idx2name[start:end] = [name] * (end - start)
            idx2local[start:end] = np.arange(0, end - start, dtype=int)
            start = end

        return idx2name, idx2local

    def get_deriv_val(self, output_name, slc, doutputs, distrib):
        deriv_val = out_idxs = None
        sz = slc.stop - slc.start
        if output_name in doutputs._views_flat:
            deriv_val = doutputs._views_flat[output_name]
            if sz != deriv_val.size:
                deriv_val = deriv_val[self.output_meta[output_name]['indices']]

        if self.nproc > 1 and not distrib:
            root = self.owning_ranks[output_name]
            if deriv_val is None:
                deriv_val = np.empty(sz)
            self.comm.Bcast(deriv_val, root=root)

        return deriv_val

    #
    # outer loop iteration functions
    #
    def single_index_iter(self, idxs):
        """
        Iterate over indices for the single index (the default) case.
        """
        for i in idxs:
            yield i, self.single_input_setter, self.single_jac_setter

    def simul_coloring_iter(self, coloring_info):
        """
        Iterate over index lists for the simul coloring case.
        """
        col_lists, _, _ = coloring_info

        for i, ilist in enumerate(col_lists):
            if i == 0:  # first outer loop give all non-colored indices.
                for j in ilist:
                    # do all non-colored indices individually
                    yield j, self.single_input_setter, self.single_jac_setter
            else:
                # yield all indices for a color at once
                yield ilist, self.array_input_setter, self.simul_coloring_jac_setter

    def par_deriv_iter(self, idxs):
        """
        Iterate over index lists for the parallel deriv case.
        """
        for tup in zip(*idxs):
            yield tup, self.array_input_setter, self.array_jac_setter

    def matmat_iter(self, idxs):
        """
        Iterate over index lists for the matrix matrix case.
        """
        for idx_list in idxs:
            yield idx_list, self.matmat_input_setter, self.matmat_jac_setter

    def par_deriv_matmat_iter(self, idxs):
        """
        Iterate over index lists for the combined parallel deriv matrix matrix case.
        """
        yield idxs, self.par_deriv_matmat_input_setter, self.par_deriv_matmat_jac_setter

    #
    # input setter functions
    #
    def single_input_setter(self, idx):
        """
        Sets 1's into the input vector in the single index case.
        """
        input_vec = self.input_vec
        input_idx_map = self.input_idx_map
        input_loc_idxs = self.input_loc_idxs

        input_name, vecname, rel_systems = input_idx_map[idx]
        dinputs = input_vec[vecname]

        loc_idx = input_loc_idxs[idx]
        if loc_idx != -1:
            dinputs._views_flat[input_name][loc_idx] = 1.0

        return rel_systems

    def array_input_setter(self, inds):
        """
        Sets 1's into the input vector in the multiple index case.
        """
        all_rel_systems = set()

        for i in inds:
            rel_systems = self.single_input_setter(i)
            _update_rel_systems(all_rel_systems, rel_systems)

        return all_rel_systems

    def matmat_input_setter(self, inds):
        """
        Sets 1's into the input vector in the matrix matrix case.
        """
        input_vec = self.input_vec
        input_idx_map = self.input_idx_map
        input_loc_idxs = self.input_loc_idxs

        input_name, vecname, rel_systems = input_idx_map[inds[0]]

        dinputs = input_vec[vecname]

        for col, i in enumerate(inds):
            loc_idx = input_loc_idxs[i]
            if loc_idx != -1:
                dinputs._views_flat[input_name][loc_idx, col] = 1.0

        return rel_systems

    def par_deriv_matmat_input_setter(self, inds):
        """
        Sets 1's into the input vector in the matrix matrix with parallel deriv case.
        """
        input_vec = self.input_vec
        input_idx_map = self.input_idx_map
        input_loc_idxs = self.input_loc_idxs

        all_rel_systems = set()

        for matmat_idxs in inds:
            input_name, vecname, rel_systems = input_idx_map[matmat_idxs[0]]
            _update_rel_systems(all_rel_systems, rel_systems)

            dinputs = input_vec[vecname]
            ncol = dinputs._ncol

            for col, i in enumerate(matmat_idxs):
                loc_idx = input_loc_idxs[i]
                if loc_idx != -1:
                    if ncol > 1:
                        dinputs._views_flat[input_name][loc_idx, col] = 1.0
                    else:
                        dinputs._views_flat[input_name][loc_idx] = 1.0


        return all_rel_systems

    #
    # Jacobian setter functions
    #
    def single_jac_setter(self, i):
        """
        Set the appropriate part of the total jacobian for a single input index.
        """
        input_name, vecname, _ = self.input_idx_map[i]
        doutputs = self.output_vec[vecname]
        relevant = self.relevant
        output_slice_map = self.output_slice_map
        fwd = self.fwd
        J = self.J

        for output_name in self.output_list:
            if input_name not in relevant or output_name in relevant[input_name]:
                slc = output_slice_map[output_name]
                deriv_val = self.get_deriv_val(output_name, slc, doutputs,
                                               self.abs2meta[output_name]['distributed'])
                if fwd:
                    J[slc, i] = deriv_val
                else:
                    J[i, slc] = deriv_val

    def array_jac_setter(self, inds):
        """
        Set the appropriate part of the total jacobian for multiple input indices.
        """
        for i in inds:
            self.single_jac_setter(i)

    def simul_coloring_jac_setter(self, inds):
        """
        Set the appropriate part of the total jacobian for simul coloring input indices.
        """
        _, row_map, _ = self.simul_coloring
        relevant = self.relevant
        output_meta = self.output_meta
        idx2local = self.idx2local
        J = self.J

        for i in inds:
            input_name, vecname, _ = self.input_idx_map[i]
            doutputs = self.output_vec[vecname]
            for row in row_map[i]:
                output_name = self.idx2name[row]
                deriv_val = out_idxs = None
                if input_name not in relevant or output_name in relevant[input_name]:
                    if output_name in doutputs._views_flat:
                        deriv_val = doutputs._views_flat[output_name]
                        if output_name in output_meta:
                            out_idxs = output_meta[output_name]['indices']
                            if out_idxs is not None:
                                deriv_val = deriv_val[out_idxs]
                        J[row, i] = deriv_val[idx2local[row]]

    def matmat_jac_setter(self, inds):
        """
        Set the appropriate part of the total jacobian for matrix matrix input indices.
        """

        # in plain matmat, all inds are for a single variable for each iteration of the outer loop,
        # so any relevance can be determined only once.
        input_name, vecname, _ = self.input_idx_map[inds[0]]
        dinputs = self.input_vec[vecname]
        doutputs = self.output_vec[vecname]
        ncol = dinputs._ncol
        output_slice_map = self.output_slice_map
        relevant = self.relevant
        nproc = self.nproc
        abs2meta = self.abs2meta
        fwd = self.fwd
        J = self.J
        owning_ranks = self.owning_ranks

        for output_name in self.output_list:
            slc = output_slice_map[output_name]
            sz = slc.stop - slc.start
            deriv_val = out_idxs = None
            if input_name not in relevant or output_name in relevant[input_name]:
                if output_name in doutputs._views_flat:
                    deriv_val = doutputs._views_flat[output_name]
                    if ncol > 1:
                        if sz != deriv_val.size // deriv_val.shape[1]:
                            deriv_val = deriv_val[output_meta[output_name]['indices']]
                    elif sz != deriv_val.size:
                        deriv_val = deriv_val[output_meta[output_name]['indices']]

                    if nproc > 1 and not abs2meta[output_name]['distributed']:
                        root = owning_ranks[output_name]
                        if deriv_val is None:
                            deriv_val = np.empty((sz, ncol))
                        self.comm.Bcast(deriv_val, root=root)

                    if fwd:
                        J[slc, inds] = deriv_val
                    else:
                        J[inds, slc] = deriv_val.T


    def par_deriv_matmat_jac_setter(self, inds):
        """
        Set the appropriate part of the total jacobian for par_deriv matrix matrix input indices.
        """
        for matmat_idxs in inds:
            self.matmat_jac_setter(matmat_idxs)


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
    """
    if all_rel_systems is _contains_all or rel_systems is _contains_all:
        all_rel_systems = _contains_all
    else:
        all_rel_systems.update(rel_systems)
