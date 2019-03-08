"""Base class used to define the interface for derivative approximation schemes."""
from __future__ import print_function, division

from itertools import groupby
from collections import defaultdict, OrderedDict
import numpy as np
from openmdao.utils.array_utils import sub2full_indices, get_local_offset_map, var_name_idx_iter, \
    update_sizes, get_input_idx_split, _get_jac_slice_dict
from openmdao.utils.name_maps import rel_name2abs_name
from openmdao.utils.coloring import color_iterator

_full_slice = slice(None)


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.

    Attributes
    ----------
    _approx_groups : list
        A list of approximation tuples ordered into groups of 'of's matching the same 'wrt'.
    _approx_groups_cached_under_cs : bool
        Flag indicates whether approx_groups was generated under complex step from higher in the
        model hieararchy.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        self._approx_groups = None
        self._approx_groups_cached_under_cs = False

    def _get_approx_groups(self, system, under_cs=False):
        """
        Retrieve data structure that contains all the approximations.

        This data structure is regenerated if we transition to or from being under a complex step
        from higher in the model hierarchy.

        Parameters
        ----------
        system : <System>
            Group or component instance.
        under_cs : bool
            Flag that indicates if we are under complex step.

        Returns
        -------
        Tuple
            Contains wrt, deltas, coeffs, current_coeff, in_idx, in_size, outputs.
        """
        if self._approx_groups is None or under_cs != self._approx_groups_cached_under_cs:
            self._init_approximations(system)
            self._approx_groups_cached_under_cs = under_cs
        return self._approx_groups

    def _update_coloring(self, system, coloring):
        """
        Replace all 'colored' approx entries with a single entry containing the actual coloring.

        Parameters
        ----------
        system : System
            The System whose approximation schemes are being updated.
        coloring : dict
            dict['fwd'] = (col_lists, row_maps)
                col_lists is a list of column lists, the first being a list of uncolored columns.
                row_maps is a list of nonzero rows for each column, or None for uncolored columns.
            dict['rev'] = (row_lists, col_maps)
                row_lists is a list of row lists, the first being a list of uncolored rows.
                col_maps is a list of nonzero cols for each row, or None for uncolored rows.
        """
        new_list = []
        new_entry = None
        colored = set()
        for tup in self._exec_list:
            key = tuple(tup[:2])
            if 'coloring' in tup[2]:
                colored.add(key)
                _, _, options = tup
                if new_entry is None:
                    options = options.copy()
                    options['coloring'] = coloring
                    options['approxs'] = [tup]
                    new_entry = (None, None, options)
                    new_list.append(new_entry)
                else:
                    new_entry[2]['approxs'].append(tup)
            elif key not in colored:
                new_list.append(tup)

        self._exec_list = new_list
        self._approx_groups = None

        # rebuid the approx_groups
        self._get_approx_groups(system)

    def add_approximation(self, abs_key, kwargs):
        """
        Use this approximation scheme to approximate the derivative d(of)/d(wrt).

        Parameters
        ----------
        abs_key : tuple(str,str)
            Absolute name pairing of (of, wrt) for the derivative.
        kwargs : dict
            Additional keyword arguments, to be interpreted by sub-classes.
        """
        raise NotImplementedError()

    def compute_approximations(self, system, jac=None, total=False):
        """
        Execute the system to compute the approximate (sub)-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        jac : None or dict-like
            If None, update system with the approximated sub-Jacobians. Otherwise, store the
            approximations in the given dict-like object.
        total : bool
            If True total derivatives are being approximated, else partials.
        """
        raise NotImplementedError()

    def _init_approximations(self, system):
        """
        Prepare for later approximations.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        global _full_slice
        from openmdao.core.group import Group
        is_total = isinstance(system, Group)

        # itertools.groupby works like `uniq` rather than the SQL query, meaning that it will only
        # group adjacent items with identical keys.
        self._exec_list.sort(key=self._key_fun)

        # groupby (along with this key function) will group all 'of's that have the same wrt and
        # step size.
        # Note: Since access to `approximations` is required multiple times, we need to
        # throw it in a list. The groupby iterator only works once.
        approx_groups = [(key, list(approx)) for key, approx in groupby(self._exec_list,
                                                                        self._key_fun)]

        outputs = system._outputs
        inputs = system._inputs
        iproc = system.comm.rank

        out_slices = outputs.get_slice_dict()
        in_slices = inputs.get_slice_dict()
        # wrt_out_offsets = get_local_offset_map(system._var_allprocs_abs_names['output'],
        #                                        system._var_sizes['nonlinear']['output'][iproc])
        # wrt_in_offsets = get_local_offset_map(system._var_allprocs_abs_names['input'],
        #                                       system._var_sizes['nonlinear']['input'][iproc])

        approx_of_idx = system._owns_approx_of_idx
        approx_wrt_idx = system._owns_approx_wrt_idx

        self._approx_groups = []
        for key, approx in approx_groups:
            wrt = key[0]
            directional = key[-1]
            data = self._get_approx_data(system, key)

            if wrt == '@color':   # use coloring (there should be only 1 of these)
                wrt_matches = system._approx_coloring_info[0]
                options = approx[0][2]
                colored_wrts = options['coloring_wrts']
                if is_total:
                    of_names = [n for n in system._var_allprocs_abs_names['output']
                                if n in system._owns_approx_of]
                    wrt_names = full_wrts = [n for n in system._var_allprocs_abs_names['output']
                                             if n in system._owns_approx_wrt]
                    ofsizes = [outputs._views_flat[of].size for of in of_names]
                    wrtsizes = [outputs._views_flat[wrt].size for wrt in wrt_names]
                    total_sizes = system._var_sizes['nonlinear']['output'][iproc]
                else:
                    of_names, wrt_names = system._get_partials_varlists()
                    ofsizes, wrtsizes = system._get_partials_sizes()
                    full_wrts = wrt_names
                    wrt_names = [rel_name2abs_name(system, n) for n in wrt_names]
                    of_names = [rel_name2abs_name(system, n) for n in of_names]
                    full_wrts = [rel_name2abs_name(system, n) for n in full_wrts]

                full_sizes = wrtsizes
                is_implicit = not is_total and \
                    system._var_sizes['nonlinear']['input'][iproc].size < wrtsizes.size
                full_ofs = list(system._outputs._views)

                if len(wrt_names) != len(wrt_matches):
                    new_names = []
                    new_sizes = []
                    for name, size in zip(wrt_names, wrtsizes):
                        if name in wrt_matches:
                            new_names.append(name)
                            new_sizes.append(size)
                    wrt_names = new_names
                    wrtsizes = new_sizes

                coloring = options['coloring']
                tmpJ = {
                    '@nrows': coloring['nrows'],
                    '@ncols': coloring['ncols'],
                    '@matrix': None,
                    '@out_slices': out_slices,
                }

                # FIXME: need to deal with mix of local/remote indices

                reduced_wrt_sizes = update_sizes(wrt_names, wrtsizes, approx_wrt_idx)
                reduced_of_sizes = update_sizes(of_names, ofsizes, approx_of_idx)
                tmpJ['@jac_slices'] = _get_jac_slice_dict(of_names, reduced_of_sizes,
                                                          wrt_names, reduced_wrt_sizes)

                if len(full_wrts) != len(colored_wrts) or approx_wrt_idx:
                    # need mapping from coloring jac columns (subset) to full jac columns
                    col_map = sub2full_indices(full_wrts, colored_wrts, full_sizes, approx_wrt_idx)
                else:
                    col_map = None

                if is_total and (approx_of_idx or len(full_ofs) > len(of_names)):
                    tmpJ['@row_idx_map'] = sub2full_indices(full_ofs, system._owns_approx_of,
                                                            total_sizes, approx_of_idx)

                for cols, nzrows in color_iterator(coloring, 'fwd'):
                    ccols = cols if col_map is None else col_map[cols]
                    idx_info = get_input_idx_split(ccols, inputs, outputs, is_implicit, is_total)
                    self._approx_groups.append((None, data, cols, tmpJ, idx_info, nzrows))
            else:
                if wrt in inputs._views_flat:
                    arr = inputs
                    slices = in_slices
                elif wrt in outputs._views_flat:
                    arr = outputs
                    slices = out_slices
                else:  # wrt is remote
                    arr = None

                if wrt in system._owns_approx_wrt_idx:
                    in_idx = np.asarray(system._owns_approx_wrt_idx[wrt], dtype=int)
                    if arr is not None:
                        in_idx += slices[wrt].start
                    in_size = len(in_idx)
                else:
                    in_size = system._var_allprocs_abs2meta[wrt]['size']
                    if arr is None:
                        in_idx = range(in_size)
                    else:
                        in_idx = range(slices[wrt].start, slices[wrt].start + in_size)

                # Directional derivatives for quick partial checking.
                # We place the indices in a list so that they are all stepped at the same time.
                if directional:
                    in_idx = [list(in_idx)]
                    in_size = 1

                tmpJ = _get_wrt_subjacs(system, approx)
                tmpJ['@out_slices'] = out_slices

                self._approx_groups.append((wrt, data, in_idx, tmpJ, [(arr, in_idx)], None))


def _gather_jac_results(comm, results):
    new_results = defaultdict(list)

    # create full results list
    all_results = comm.allgather(results)
    for proc_results in all_results:
        for key in proc_results:
            new_results[key].extend(proc_results[key])

    return new_results


def _get_wrt_subjacs(system, approxs):
    """
    Return a dict mapping wrt names to contiguous memory views of all of their nonzero subjacs.

    All nonzero subjacs for a particular wrt are 'compressed' together so they're contiguous.

    This allows for setting an entire column of the jacobian at once instead of looping over
    each subjac.
    """
    abs_out_names = system._var_allprocs_abs_names['output']
    abs2idx = system._var_allprocs_abs2idx['nonlinear']
    abs2meta = system._var_allprocs_abs2meta
    approx_of_idx = system._owns_approx_of_idx
    approx_wrt_idx = system._owns_approx_wrt_idx
    iproc = system.comm.rank

    J = {}
    ofdict = {}

    # in the non-colored case, all wrts will be the same for all entries in approxs
    for of, wrt, options in approxs:
        if wrt not in J:
            J[wrt] = {'ofs': [], 'tot_rows': 0, 'directional': options['directional']}

        if of not in ofdict:
            J[wrt]['ofs'].append(of)
            if of in approx_of_idx:
                out_idx = approx_of_idx[of]
                out_size = len(out_idx)
            else:
                out_size = abs2meta[of]['size']
                out_idx = _full_slice
            ofdict[of] = (out_size, out_idx)
            J[wrt]['tot_rows'] += out_size

    for wrt in J:
        lst = J[wrt]['ofs']
        J[wrt]['ofs'] = wrt_ofs = OrderedDict()

        # create dense array to contain all nonzero subjacs for this wrt
        if J[wrt]['directional']:
            J[wrt]['data'] = arr = np.zeros((J[wrt]['tot_rows'], 1))
        elif wrt in approx_wrt_idx:
            J[wrt]['data'] = arr = np.zeros((J[wrt]['tot_rows'], len(approx_wrt_idx[wrt])))
        else:
            J[wrt]['data'] = arr = np.zeros((J[wrt]['tot_rows'], abs2meta[wrt]['size']))

        # sort ofs into the proper order to match outputs/resids vecs
        start = end = 0
        sorted_ofs = sorted(lst, key=lambda n: abs2idx[n])
        for of in sorted_ofs:
            osize, oidx = ofdict[of]
            end += osize
            # store subview corresponding to the (of, wrt) subjac and any index info
            wrt_ofs[of] = (arr[start:end, :], oidx)
            start = end

        if len(sorted_ofs) != len(system._var_allprocs_abs_names['output']):
            ofset = set(sorted_ofs)
            J[wrt]['full_out_idxs'] = \
                sub2full_indices(system._var_allprocs_abs_names['output'], ofset,
                                 system._var_sizes['nonlinear']['output'][iproc],
                                 approx_of_idx)
        else:
            J[wrt]['full_out_idxs'] = _full_slice

    return J
