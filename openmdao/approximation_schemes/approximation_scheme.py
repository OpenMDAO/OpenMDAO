"""Base class used to define the interface for derivative approximation schemes."""
from __future__ import print_function, division

from collections import defaultdict, OrderedDict
import numpy as np
from openmdao.utils.array_utils import sub2full_indices, update_sizes

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
        Perform any necessary setup for the approximation scheme.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        pass


def _gather_jac_results(comm, results):
    new_results = defaultdict(list)

    # create full results list
    all_results = comm.allgather(results)
    for proc_results in all_results:
        for key in proc_results:
            new_results[key].extend(proc_results[key])

    return new_results


def _get_jac_slice_dict(of_names, of_sizes, wrt_names, wrt_sizes):
    """
    Return a dict of (of,wrt) pairs mapped to slices of a dense matrix.
    """
    dct = {}
    rstart = rend = 0
    for ofname, ofsize in zip(of_names, of_sizes):
        rend += ofsize
        cstart = cend = 0
        for wrtname, wrtsize in zip(wrt_names, wrt_sizes):
            cend += wrtsize
            dct[(ofname, wrtname)] = (slice(rstart, rend), slice(cstart, cend))
            cstart = cend
        rstart = rend
    return dct


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
            J[wrt] = {'ofs': [], 'tot_rows': 0}

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
        if wrt in approx_wrt_idx:
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
