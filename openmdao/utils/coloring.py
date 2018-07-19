"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import os
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import combinations
from distutils.version import LooseVersion

from six import iteritems
from six.moves import range

import numpy as np
from numpy.random import rand

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.matrix import sparse_types
from openmdao.utils.array_utils import array_viz

# If this is True, then IF simul coloring/sparsity is specified, use it.
# If False, don't use it regardless.
# The command line simul_coloring and sparsity commands make this False when generating a
# new coloring and/or sparsity.
_use_sparsity = True


# numpy versions before 1.12 don't use the 'axis' arg passed to count_nonzero and always
# return an int instead of an array of ints, so create our own function for those versions.
if LooseVersion(np.__version__) >= LooseVersion("1.12"):
    _count_nonzeros = np.count_nonzero
else:
    def _count_nonzeros(arr, axis=None):
        if axis == 1:  # rows
            count = np.empty(arr.shape[0], dtype=int)
            for row in range(arr.shape[0]):
                count[row] = np.count_nonzero(arr[row])
        elif axis == 0:  # cols
            count = np.empty(arr.shape[1], dtype=int)
            for col in range(arr.shape[1]):
                count[col] = np.count_nonzero(arr[:, col])
        else:
            return np.count_nonzero(arr)

        return count


class _SubjacRandomizer(object):
    """
    A replacement for Jacobian._set_abs that replaces subjac with random numbers.

    Attributes
    ----------
    _orig_set_abs : bound function
        Original _set_abs function for the given Jacobian.
    _jac : Jacobian
        The jacobian having its _set_abs replaced.
    _tol : float
        Tolerance used to shift random numbers away from 0.
    """

    def __init__(self, jac, tol):
        """
        Initialize the function replacement.

        Parameters
        ----------
        jac : Jacobian
            The Jacobian having its _set_abs method replaced.
        tol : float
            Values between -tol and tol will be shifted away from 0.
        """
        self._orig_set_abs = jac._set_abs
        self._jac = jac
        self._tol = tol

    def __call__(self, key, subjac):
        """
        Call this function replacement.

        Parameters
        ----------
        key : (str, str)
            Tuple of (response_name, dv_name)
        subjac : array-like
            Value of the subjacobian being assigned to key.
        """
        jac = self._jac
        tol = self._tol

        if key in jac._subjacs_info:
            info = jac._subjacs_info[key]
            rows = info['rows']
        else:
            rows = None

        if rows is not None:  # list form
            subjac = rand(rows.size) + 1.0
        elif isinstance(subjac, sparse_types):  # sparse
            subjac = subjac.copy()
            subjac.data = rand(subjac.data.size) + 1.0
        elif isinstance(subjac, np.ndarray):   # dense array
            subjac = rand(*(subjac.shape)) + 1.0
        else:  # scalar
            subjac = rand() + 1.0

        self._orig_set_abs(key, subjac)


def _order_by_ID(col_matrix):
    """
    Return columns in order of incidence degree (ID).

    ID is the number of already colored neighbors (neighbors are dependent columns).

    Parameters
    ----------
    col_matrix : ndarray
        Boolean array of column dependencies.

    Yields
    ------
    int
        Column index.
    """
    degrees = _count_nonzeros(col_matrix, axis=0)
    ncols = degrees.size

    # use max degree column as a starting point instead of just choosing a random column
    # since all have incidence degree of 0 when we start.
    start = degrees.argmax()
    yield start

    colored_degrees = np.zeros(degrees.size, dtype=int)
    colored_degrees[col_matrix[start]] += 1
    colored_degrees[start] = -ncols  # ensure that this col will never have max degree again

    for i in range(ncols - 1):
        col = colored_degrees.argmax()
        colored_degrees[col_matrix[col]] += 1
        colored_degrees[col] = -ncols  # ensure that this col will never have max degree again
        yield col


def _J2col_matrix(J):
    """
    Convert boolean jacobian sparsity matrix to a column adjacency matrix.

    Parameters
    ----------
    J : ndarray
        Boolean jacobian sparsity matrix.

    Returns
    -------
    ndarray
        Column adjacency matrix.
    """
    nrows, ncols = J.shape

    col_matrix = np.zeros((ncols, ncols), dtype=bool)

    # mark col_matrix entries as True when nonzero row entries make them dependent
    for row in range(nrows):
        nzro = np.nonzero(J[row])[0]
        for col in nzro:
            col_matrix[col, nzro] = True
            col_matrix[nzro, col] = True

    # zero out diagonal (column is not adjacent to itself)
    np.fill_diagonal(col_matrix, False)

    return col_matrix


def _get_full_disjoint_cols(J):
    """
    Find sets of disjoint columns in J and their corresponding rows using a col adjacency matrix.

    Parameters
    ----------
    J : ndarray
        The total jacobian.

    Returns
    -------
    list
        List of lists of disjoint columns
    """
    color_groups = []
    nrows, ncols = J.shape
    col_matrix = _J2col_matrix(J)

    # -1 indicates that a column has not been colored
    colors = np.full(ncols, -1, dtype=int)

    for col in _order_by_ID(col_matrix):
        neighbor_colors = set(colors[col_matrix[col]])
        for color, grp in enumerate(color_groups):
            if color not in neighbor_colors:
                grp.add(col)
                colors[col] = color
                break
        else:
            colors[col] = len(color_groups)
            color_groups.append(set([col]))

    return color_groups


def _get_full_disjoint_bipartite(J):
    """
    Find disjoint column and row sets in J and their corresponding rows/cols.

    This finds a complete direct cover of the edges of the bipartite graph defined by J.
    Each nonzero in J is an edge in the graph.  Each row and column in J is a vertex in
    the graph.

    Parameters
    ----------
    J : ndarray
        The total jacobian.

    Returns
    -------
    list
        List of lists of disjoint columns
    """
    nrows, ncols = J.shape

    row_groups = []
    column_groups = []

    nonzero_rows = [None] * ncols
    nonzero_cols = [None] * nrows

    # use this to keep track of the current set of nonzeros as they are removed during coloring
    current_row_nz = [set(np.nonzero(J[:, c])[0]) for c in range(ncols)]
    current_col_nz = [set(np.nonzero(row)[0]) for row in J]

    row_degrees = _count_nonzeros(J, axis=1)  # row degrees
    col_degrees = _count_nonzeros(J, axis=0)  # column degrees

    uncolored_row_idxs = np.argsort(row_degrees)[::-1]
    uncolored_col_idxs = np.argsort(col_degrees)[::-1]

    uncolored_row_deg = row_degrees[uncolored_row_idxs]
    uncolored_col_deg = col_degrees[uncolored_col_idxs]

    # each nonzero entry in J is an edge in the bipartite graph.  We have to make sure we
    # cover every edge.
    num_edges = _count_nonzeros(J)
    edge_count = 0

    while edge_count < num_edges:

        if uncolored_col_deg[0] >= uncolored_row_deg[0]:  # choose max deg column
            max_c = uncolored_col_idxs[0]
            uncolored_col_deg[0] = -1
            color_group = set([max_c])
            nonzero_rows[max_c] = list(current_row_nz[max_c])
            nz_to_remove = current_row_nz[max_c]
            current_row_nz[max_c] = set()  # free up memory

            for i, w in enumerate(uncolored_col_idxs[1:]):
                nz = current_row_nz[w]
                # add to group if not connected to existing group via path of length 2
                for r in nz:
                    if not color_group.isdisjoint(current_col_nz[r]):
                        break
                else:
                    color_group.add(w)
                    uncolored_col_deg[i + 1] = -1
                    nonzero_rows[w] = list(nz)
                    nz_to_remove.update(nz)
                    for r in nz:
                        row_degrees[r] -= 1

            column_groups.append(color_group)

            # remove nonzeros in colored columns
            edge_count += len(nz_to_remove)
            for r in nz_to_remove:
                current_col_nz[r] -= color_group

            uncolored_col_idxs = uncolored_col_idxs[uncolored_col_deg > 0]
            uncolored_col_deg = col_degrees[uncolored_col_idxs]

            # update and resort the uncolored row degrees
            uncolored_row_deg = row_degrees[uncolored_row_idxs]
            sorting = np.argsort(uncolored_row_deg)[::-1]
            uncolored_row_idxs = uncolored_row_idxs[sorting]
            uncolored_row_deg = uncolored_row_deg[sorting]

        else:  # choose max deg row
            max_r = uncolored_row_idxs[0]
            uncolored_row_deg[0] = -1

            color_group = set([max_r])
            nonzero_cols[max_r] = list(current_col_nz[max_r])
            nz_to_remove = current_col_nz[max_r]
            current_col_nz[max_r] = set()  # free up memory

            for i, w in enumerate(uncolored_row_idxs[1:]):
                nz = current_col_nz[w]
                # add to group if not connected to existing group via path of length 2
                for c in nz:
                    if not color_group.isdisjoint(current_row_nz[c]):
                        break
                else:
                    color_group.add(w)
                    uncolored_row_deg[i + 1] = -1
                    nonzero_cols[w] = list(nz)
                    nz_to_remove.update(nz)
                    for c in nz:
                        col_degrees[c] -= 1

            row_groups.append(color_group)

            # remove nonzeros in colored columns
            edge_count += len(nz_to_remove)
            for c in nz_to_remove:
                current_row_nz[c] -= color_group

            uncolored_row_idxs = uncolored_row_idxs[uncolored_row_deg > 0]
            uncolored_row_deg = row_degrees[uncolored_row_idxs]

            # update and resort the uncolored column degrees
            uncolored_col_deg = col_degrees[uncolored_col_idxs]
            sorting = np.argsort(uncolored_col_deg)[::-1]
            uncolored_col_idxs = uncolored_col_idxs[sorting]
            uncolored_col_deg = uncolored_col_deg[sorting]

    return column_groups, row_groups, nonzero_rows, nonzero_cols


def _tol_sweep(arr, tol=1e-15, orders=5):
    """
    Find best tolerance 'around' tol to choose nonzero values of arr.

    # Sweeps over tolerances +- 'orders' orders of magnitude around tol and picks the most
    # stable one (one corresponding to the most repeated number of nonzero entries).

    Parameters
    ----------
    arr : ndarray
        The array requiring computation of nonzero values.
    tol : float
        Tolerance.  We'll sweep above and below this by 'orders' of magnitude.
    orders : int
        Number of orders of magnitude for one direction of our sweep.

    Returns
    -------
    float
        Chosen tolerance.
    int
        Number of repeated nonzero counts for the given tolerance.
    int
        Number of tolerances tested in the sweep.
    int
        Number of zero entries at chosen tolerance.
    """
    nzeros = defaultdict(list)
    itol = tol * 10.**orders
    smallest = tol / 10.**orders
    n_tested = 0
    while itol >= smallest:
        if itol < 1.:
            num_zero = arr[arr <= itol].size
            nzeros[num_zero].append(itol)
            n_tested += 1
        itol /= 10.

    # pick lowest tolerance corresponding to the most repeated number of 'zero' entries
    sorted_items = sorted(nzeros.items(), key=lambda x: len(x[1]), reverse=True)
    good_tol = sorted_items[0][1][-1]

    return good_tol, len(sorted_items[0][1]), n_tested, sorted_items[0][0]


def _get_bool_jac(prob, repeats=3, tol=1e-15, orders=5, setup=False, run_model=False):
    """
    Return a boolean version of the total jacobian.

    The jacobian is computed by calculating a total jacobian using _compute_totals 'repeats'
    times and adding the absolute values of those together, then dividing by the max value,
    then converting to a boolean array, specifying all entries below a tolerance as False and all
    others as True.  Prior to calling _compute_totals, all of the partial jacobians in the
    model are modified so that when any of their subjacobians are assigned a value, that
    value is populated with positive random numbers in the range [1.0, 2.0).

    Parameters
    ----------
    prob : Problem
        The Problem being analyzed.
    repeats : int
        Number of times to repeat total jacobian computation.
    tol : float
        Starting tolerance on values in jacobian.  Actual tolerance is computed based on
        consistent numbers of zero entries over a sweep of tolerances.  Anything smaller in
        magnitude than the computed tolerance will be set to 0.0.
    orders : int
        Number of orders of magnitude for up and down tolerance sweep (default is 5).
    setup : bool
        If True, run setup before calling compute_totals.
    run_model : bool
        If True, run run_model before calling compute_totals.

    Returns
    -------
    ndarray
        A boolean composite of 'repeats' total jacobians.
    """
    # clear out any old simul coloring info
    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    if setup:
        prob.setup(mode=prob._mode)

    if run_model:
        prob.run_model()

    seen = set()
    for system in prob.model.system_iter(recurse=True, include_self=True):
        jac = system._assembled_jac
        if jac is None:
            jac = system._jacobian
        if jac is not None and jac not in seen:
            # replace jacobian set_abs with one that replaces all subjacs with random numbers
            jac._set_abs = _SubjacRandomizer(jac, tol)
            seen.add(jac)

    wrt = list(prob.driver._designvars)

    # get responses in order used by the driver
    of = prob.driver._get_ordered_nl_responses()

    if not of or not wrt:
        raise RuntimeError("Sparsity structure cannot be computed without declaration of design "
                           "variables and responses.")

    start_time = time.time()
    fullJ = None
    for i in range(repeats):
        J = prob.driver._compute_totals(return_format='array', of=of, wrt=wrt)
        if fullJ is None:
            fullJ = np.abs(J)
        else:
            fullJ += np.abs(J)
    elapsed = time.time() - start_time

    # normalize the full J by dividing by the max value
    fullJ /= np.max(fullJ)

    good_tol, nz_matches, n_tested, zero_entries = _tol_sweep(fullJ, tol, orders)

    print("\nUsing tolerance: %g" % good_tol)
    print("Most common number of zero entries (%d of %d) repeated %d times out of %d tolerances "
          "tested.\n" % (zero_entries, fullJ.size, nz_matches, n_tested))
    print("Full total jacobian was computed %d times, taking %f seconds." % (repeats, elapsed))
    print("Total jacobian shape:", fullJ.shape, "\n")

    # now revert the _jacobian _set_abs methods back to their original values
    seen = set()
    for system in prob.model.system_iter(recurse=True, include_self=True):
        jac = system._assembled_jac
        if jac is None:
            jac = system._jacobian
        if jac is not None and jac not in seen:
            randomizer = jac._set_abs
            jac._set_abs = randomizer._orig_set_abs
            seen.add(jac)

    boolJ = np.zeros(fullJ.shape, dtype=bool)
    boolJ[fullJ > good_tol] = True

    return boolJ


def _sparsity_from_jac(J, of, wrt, driver):
    """
    Given a boolean total jacobian and a driver, compute subjac sparsity.

    Parameters
    ----------
    J : ndarray
        Boolean total jacobian.
    of : list of str
        List of responses.
    wrt : list of str
        List of design vars.
    driver : <Driver>
        Driver containing responses and design variables.

    Returns
    -------
    OrderedDict
        Nested OrderedDict of form sparsity[response][desvar] = (rows, cols, shape)
    """
    sparsity = OrderedDict()
    row_start = row_end = 0
    res_meta = driver._responses
    dv_meta = driver._designvars

    for res in of:
        sparsity[res] = OrderedDict()
        res_size = int(res_meta[res]['size'])
        row_end += res_size
        col_start = col_end = 0
        for dv in wrt:
            dv_size = int(dv_meta[dv]['size'])
            col_end += dv_size

            # save sparsity structure as  (rows, cols, shape)
            irows, icols = np.nonzero(J[row_start:row_end, col_start:col_end])

            # convert to make JSON serializable
            irows = [int(i) for i in irows]
            icols = [int(i) for i in icols]
            sparsity[res][dv] = (irows, icols, (res_size, dv_size))

            col_start = col_end

        row_start = row_end

    return sparsity


def _write_sparsity(sparsity, stream):
    """
    Write the sparsity structure to the given stream.

    Parameters
    ----------
    sparsity : dict
        Nested dict of subjac sparsity for each total derivative.
    stream : file-like
        Output stream.
    """
    stream.write("{\n")

    last_res_idx = len(sparsity) - 1
    for i, (out, out_dict) in enumerate(iteritems(sparsity)):
        stream.write('"%s": {\n' % out)
        last_dv_idx = len(out_dict) - 1
        for j, (inp, subjac) in enumerate(iteritems(out_dict)):
            rows, cols, shape = subjac
            if len(rows) > 15:
                stream.write('   "%s": [\n' % inp)
                stream.write('        %s,\n' % rows)
                stream.write('        %s,\n' % cols)
                stream.write('        %s]' % list(shape))
            else:
                stream.write('   "%s": [%s, %s, %s]' % (inp, rows, cols, list(shape)))
            if j == last_dv_idx:
                stream.write('\n')
            else:
                stream.write(',\n')
        stream.write("}")
        if i == last_res_idx:
            stream.write('\n')
        else:
            stream.write(',\n')

    stream.write("}")


def _write_coloring(modes, color_info, stream):
    """
    Write the coloring and sparsity structures to the given stream.

    Parameters
    ----------
    modes : list of str
        Derivative direction.
    color_info : dict
        dict['fwd'] = (col_lists, row_maps)
            col_lists is a list of column lists, the first being a list of uncolored columns.
            row_maps is a list of nonzero rows for each column, or None for uncolored columns.
        dict['rev'] = (row_lists, col_maps)
            row_lists is a list of row lists, the first being a list of uncolored rows.
            col_maps is a list of nonzero cols for each row, or None for uncolored rows.
        dict['sparsity'] = a nested dict specifying subjac sparsity for each total derivative.
        dict['J'] = ndarray, the computed boolean jacobian.
    stream : file-like
        Output stream.
    """
    tty = stream.isatty()
    none = 'None' if tty else 'null'
    sparsity = color_info.get('sparsity')

    stream.write("{\n")
    for m, mode in enumerate(modes):
        name = 'column' if mode == 'fwd' else 'row'
        lists, nonzero_entries = color_info[mode]

        if m > 0:
            stream.write(",\n")

        stream.write('"%s": [[\n' % mode)
        last_idx = len(lists) - 1
        for i, lst in enumerate(lists):
            stream.write("   %s" % lst)
            if i < last_idx:
                stream.write(",")

            if tty:
                if i == 0:
                    stream.write("   # uncolored %ss" % name)
                else:
                    stream.write("   # color %d" % i)

            stream.write("\n")

        stream.write("],\n[\n")
        last_idx = len(nonzero_entries) - 1
        for i, nonzeros in enumerate(nonzero_entries):
            if nonzeros is None:
                stream.write("   %s" % none)
            else:
                stream.write("   %s" % nonzeros)

            if i < last_idx:
                stream.write(",")

            if tty:
                stream.write("   # %s %d" % (name, i))

            stream.write("\n")

        stream.write("]]")

    if sparsity:
        stream.write(',\n"sparsity": ')
        _write_sparsity(sparsity, stream)
    else:
        stream.write(',\n"sparsity": %s' % none)

    stream.write("\n}")


def get_sparsity(problem, mode='fwd', repeats=1, tol=1.e-15, show_jac=False,
                 setup=False, run_model=False, stream=sys.stdout):
    """
    Compute derivative sparsity for the given problem.

    Parameters
    ----------
    problem : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
    repeats : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance used to determine if an array entry is nonzero.
    show_jac : bool
        If True, display a visualization of the final total jacobian used to compute the coloring.
    stream : file-like or None
        Stream where output coloring info will be written.
    setup : bool
        If True, run setup before calling compute_totals.
    run_model : bool
        If True, run run_model before calling compute_totals.

    Returns
    -------
    dict
        A nested dict specifying subjac sparsity for each total deriv, e.g., sparsity[resp][dv].
    """
    driver = problem.driver

    J = _get_bool_jac(problem, repeats=repeats, tol=tol, setup=setup,
                      run_model=run_model)

    of = driver._get_ordered_nl_responses()
    wrt = list(driver._designvars)

    sparsity = _sparsity_from_jac(J, of, wrt, driver)

    driver._total_jac = None

    if stream is not None:
        _write_sparsity(sparsity, stream)
        stream.write("\n")

        if show_jac and stream is not None:
            stream.write("\n\n")
            array_viz(J, problem, of, wrt, stream)

    return sparsity


def _total_solves(color_info):
    """
    Return total number of linear solves required based on the given coloring info.

    Parameters
    ----------
    color_info : dict
        dict['fwd'] = (col_lists, row_maps)
            col_lists is a list of column lists, the first being a list of uncolored columns.
            row_maps is a list of nonzero rows for each column, or None for uncolored columns.
        dict['rev'] = (row_lists, col_maps)
            row_lists is a list of row lists, the first being a list of uncolored rows.
            col_maps is a list of nonzero cols for each row, or None for uncolored rows.
        dict['sparsity'] = a nested dict specifying subjac sparsity for each total derivative.
        dict['J'] = ndarray, the computed boolean jacobian.

    Returns
    -------
    int
        Total number of linear solves required to compute the total Jacobian.
    """
    total_solves = 0

    # lists[0] are the uncolored columns or rows, which are solved individually so
    # we add all of them, along with the number of remaining lists, where each
    # sublist is a bunch of columns or rows that are solved together, to get the total colors
    # (which equals the total number of linear solves).
    if 'fwd' in color_info:
        row_lists, _ = color_info['fwd']
        total_solves += len(row_lists[0]) + len(row_lists) - 1
    if 'rev' in color_info:
        col_lists, _ = color_info['rev']
        total_solves += len(col_lists[0]) + len(col_lists) - 1

    return total_solves


def _solves_info(color_info):
    """
    Return info about the number of colors given the current coloring scheme.

    Parameters
    ----------
    color_info : dict
        dict['fwd'] = (col_lists, row_maps)
            col_lists is a list of column lists, the first being a list of uncolored columns.
            row_maps is a list of nonzero rows for each column, or None for uncolored columns.
        dict['rev'] = (row_lists, col_maps)
            row_lists is a list of row lists, the first being a list of uncolored rows.
            col_maps is a list of nonzero cols for each row, or None for uncolored rows.
        dict['sparsity'] = a nested dict specifying subjac sparsity for each total derivative.
        dict['J'] = ndarray, the computed boolean jacobian.

    Returns
    -------
    float
        Total size (minimum chosed based on which mode is better).
    float
        Total solves.
    """
    rev_size, fwd_size = color_info['J'].shape
    tot_colors = _total_solves(color_info)

    fwd_solves = rev_solves = 0
    if tot_colors == 0:  # no coloring found
        tot_colors = tot_size = min([rev_size, fwd_size])
        pct = 0.
    else:
        fwd_lists = color_info['fwd'][0] if 'fwd' in color_info else []
        rev_lists = color_info['rev'][0] if 'rev' in color_info else []

        if fwd_lists and not rev_lists:
            tot_size = fwd_size
        elif rev_lists and not fwd_lists:
            tot_size = rev_size
        else:
            tot_size = min(fwd_size, rev_size)

        if fwd_lists:
            fwd_solves = len(fwd_lists[0]) + len(fwd_lists) - 1

        if rev_lists:
            rev_solves = len(rev_lists[0]) + len(rev_lists) - 1

        pct = ((tot_size - tot_colors) / tot_size * 100)

    return tot_size, tot_colors, fwd_solves, rev_solves, pct


def _compute_coloring(J, mode):
    """
    Compute a good coloring in a specified dominant direction.

    Parameters
    ----------
    J : ndarray
        The boolean total jacobian.
    mode : str
        The direction for solving for total derivatives.  If 'auto', use bidirectional coloring.

    Returns
    -------
    coloring_info
        dict
            dict['fwd'] = (col_lists, row_maps)
                col_lists is a list of column lists, the first being a list of uncolored columns.
                row_maps is a list of nonzero rows for each column, or None for uncolored columns.
            dict['rev'] = (row_lists, col_maps)
                row_lists is a list of row lists, the first being a list of uncolored rows.
                col_maps is a list of nonzero cols for each row, or None for uncolored rows.
            dict['sparsity'] = a nested dict specifying subjac sparsity for each total derivative.
            dict['J'] = ndarray, the computed boolean jacobian.
    """
    bidirectional = mode == 'auto'

    rev = mode == 'rev'

    ###################################
    # Bidirectional coloring algorithm
    ###################################
    #
    # See 'An Algorithm for Complete Direct Cover' in 'Computing a Sparse Jacobian Matrix by
    #     Rows and Columns' by Hossain and Steihaug.
    #
    # Note that when we're done, the coloring for the chosen direction will contain a list of
    #     column or row lists, with the first entry containing the indices of the uncolored
    #     rows or cols for that direction, and the coloring for the opposite direction will
    #     have the same format.
    #

    if bidirectional:
        col_groups, row_groups, rowcol_map, colrow_map = _get_full_disjoint_bipartite(J)
    else:
        if rev:
            J = J.T
        col_groups = _get_full_disjoint_cols(J)
        row_groups = []

    uncolored_cols = [cset.pop() for cset in col_groups if len(cset) == 1]
    col_groups = [list(cset) for cset in col_groups if len(cset) > 1]

    uncolored_rows = [rset.pop() for rset in row_groups if len(rset) == 1]
    row_groups = [list(rset) for rset in row_groups if len(rset) > 1]

    tot_colors = len(uncolored_cols) + len(uncolored_rows) + len(col_groups) + len(row_groups)

    # the first lists entry corresponds to all uncolored columns (columns that are not
    # disjoint wrt any other columns).  The other entries are groups of columns that do not
    # share any nonzero row entries in common.
    clists = [uncolored_cols]
    clists.extend(col_groups)

    # now do the same for rows
    rlists = [uncolored_rows]
    rlists.extend(row_groups)

    if not bidirectional:
        if clists:
            rowcol_map = [None] * J.shape[1]  # will contain list of nonzero rows for each column
            for clist in col_groups:
                for col in clist:
                    # ndarrays are converted to lists to be json serializable
                    rowcol_map[col] = list(np.nonzero(J[:, col])[0])
        else:
            rowcol_map = []

        if rlists:
            colrow_map = [None] * J.shape[0]  # will contain list of nonzero cols for each row
            for rlist in row_groups:
                for row in rlist:
                    # ndarrays are converted to lists to be json serializable
                    colrow_map[row] = list(np.nonzero(J[row])[0])
        else:
            colrow_map = []

    if rev:
        clists, rlists = rlists, clists
        rowcol_map, colrow_map = colrow_map, rowcol_map
        J = J.T

    coloring = {
        'fwd': [clists, rowcol_map],
        'rev': [rlists, colrow_map],
        'J': J
    }

    return coloring


def get_simul_meta(problem, mode=None, repeats=1, tol=1.e-15, show_jac=False,
                   include_sparsity=True, setup=False, run_model=False, bool_jac=None,
                   stream=sys.stdout):
    """
    Compute simultaneous derivative colorings for the given problem.

    Parameters
    ----------
    problem : Problem or None
        The Problem being analyzed.
    mode : str or None
        The direction for computing derivatives.  If None, use problem._mode.
    repeats : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance used to determine if an array entry is nonzero.
    show_jac : bool
        If True, display a visualiation of the final total jacobian used to compute the coloring.
    include_sparsity : bool
        If True, include the sparsity structure of the total jacobian mapped to design vars
        and responses.  (This info is used by pyOptSparseDriver).
    setup : bool
        If True, run setup before calling compute_totals.
    run_model : bool
        If True, run run_model before calling compute_totals.
    bool_jac : ndarray
        If problem is not supplied, a previously computed boolean jacobian can be used.
    stream : file-like or None
        Stream where output coloring info will be written.

    Returns
    -------
    dict
        dict['fwd'] = (col_lists, row_maps)
            col_lists is a list of column lists, the first being a list of uncolored columns.
            row_maps is a list of nonzero rows for each column, or None for uncolored columns.
        dict['rev'] = (row_lists, col_maps)
            row_lists is a list of row lists, the first being a list of uncolored rows.
            col_maps is a list of nonzero cols for each row, or None for uncolored rows.
        dict['sparsity'] = a nested dict specifying subjac sparsity for each total derivative.
        dict['J'] = ndarray, the computed boolean jacobian.
        dict['time_sparsity'] = float, the time to compute the sparsity matrix.
        dict['time_coloring'] = float, the time to compute the coloring, given the sparsity matrix.
    """
    sparsity = None

    if problem is not None:
        driver = problem.driver
        if mode is None:
            mode = problem._orig_mode
        if mode != problem._orig_mode:
            raise RuntimeError("given mode (%s) does not agree with Problem mode (%s)" %
                               (mode, problem._mode))
        start_time = time.time()
        J = _get_bool_jac(problem, repeats=repeats, tol=tol, setup=setup,
                          run_model=run_model)
        time_sparsity = time.time() - start_time

        if include_sparsity or (show_jac and stream is not None):
            of = driver._get_ordered_nl_responses()
            wrt = list(driver._designvars)

        if include_sparsity:
            sparsity = _sparsity_from_jac(J, of, wrt, driver)

        driver._total_jac = None
    elif bool_jac is not None:
        J = bool_jac
        time_sparsity = 0.
        if mode is None:
            mode = 'auto'
        driver = None
    else:
        raise RuntimeError("You must supply either problem or bool_jac to get_simul_meta().")

    start_time = time.time()
    coloring = _compute_coloring(J, mode)
    coloring['time_coloring'] = time.time() - start_time
    coloring['time_sparsity'] = time_sparsity

    modes = [m for m in ('fwd', 'rev') if m in coloring]

    if driver is not None and include_sparsity:
        coloring['sparsity'] = sparsity

    if stream is not None:
        if stream.isatty():
            stream.write("\n########### BEGIN COLORING DATA ################\n")
            _write_coloring(modes, coloring, stream)
            stream.write("\n########### END COLORING DATA ############\n")
        else:
            _write_coloring(modes, coloring, stream)

        if show_jac:
            s = stream if stream.isatty() else sys.stdout
            s.write("\n\n")
            array_viz(J, problem, of, wrt, s)

    return coloring


def simul_coloring_summary(color_info, stream=sys.stdout):
    """
    Print a summary of simultaneous coloring info for the given problem and coloring metadata.

    Parameters
    ----------
    color_info : dict
        Coloring metadata.
    stream : file-like
        Where the output will go.
    """
    stream.write("\nTime to compute sparsity: %f\n" % color_info.get('time_sparsity', 0.))
    stream.write("Time to compute coloring: %f\n" % color_info.get('time_coloring', 0.))

    if 'fwd' not in color_info and 'rev' not in color_info:
        tot_size = min(color_info['J'].shape)
        stream.write("\nSimultaneous derivatives can't improve on the total number of solves "
                     "required (%d) for this configuration\n" % tot_size)
    else:
        tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(color_info)

        stream.write("\nFWD solves: %d   REV solves: %d" % (fwd_solves, rev_solves))
        stream.write("\n\nTotal colors vs. total size: %d vs %d  (%.1f%% improvement)\n" %
                     (tot_colors, tot_size, pct))


def dynamic_sparsity(driver):
    """
    Compute deriv sparsity during runtime.

    Parameters
    ----------
    driver : <Driver>
        The driver performing the optimization.
    """
    problem = driver._problem
    driver._total_jac = None
    repeats = driver.options['dynamic_derivs_repeats']

    # save the sparsity.json file for later inspection
    with open("sparsity.json", "w") as f:
        sparsity = get_sparsity(problem, mode=problem._mode, repeats=repeats, stream=f)

    driver.set_total_jac_sparsity(sparsity)
    driver._setup_tot_jac_sparsity()


def dynamic_simul_coloring(driver, do_sparsity=False):
    """
    Compute simultaneous deriv coloring during runtime.

    Parameters
    ----------
    driver : <Driver>
        The driver performing the optimization.
    do_sparsity : bool
        If True, setup the total jacobian sparsity (needed by pyOptSparseDriver).
    """
    problem = driver._problem
    driver._total_jac = None

    # save the coloring.json file for later inspection
    with open("coloring.json", "w") as f:
        coloring = get_simul_meta(problem,
                                  repeats=driver.options['dynamic_derivs_repeats'],
                                  tol=1.e-15, include_sparsity=True,
                                  setup=False, run_model=False, stream=f)
    driver.set_simul_deriv_color(coloring)
    driver._setup_simul_coloring()
    if do_sparsity:
        driver._setup_tot_jac_sparsity()


def _simul_coloring_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao simul_coloring' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', action='store', dest='outfile', help='output file (json format)')
    parser.add_argument('-n', action='store', dest='num_jacs', default=3, type=int,
                        help='number of times to repeat total derivative computation')
    parser.add_argument('-t', action='store', dest='tolerance', default=1.e-15, type=float,
                        help='tolerance used to determine if a total jacobian entry is nonzero')
    parser.add_argument('-j', '--jac', action='store_true', dest='show_jac',
                        help="Display a visualization of the final total jacobian used to "
                        "compute the coloring.")
    parser.add_argument('--no-sparsity', action='store_true', dest='no_sparsity',
                        help="Exclude the sparsity structure from the coloring data structure.")
    parser.add_argument('-p', '--profile', action='store_true', dest='profile',
                        help="Do profiling on the coloring process.")


def _simul_coloring_cmd(options):
    """
    Return the post_setup hook function for 'openmdao simul_coloring'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    from openmdao.core.problem import Problem
    from openmdao.devtools.debug import profiling
    from openmdao.utils.general_utils import do_nothing_context

    global _use_sparsity

    _use_sparsity = False

    def _simul_coloring(prob):
        if options.outfile is None:
            outfile = sys.stdout
        else:
            outfile = open(options.outfile, 'w')
        Problem._post_setup_func = None  # avoid recursive loop

        with profiling('coloring_profile.out') if options.profile else do_nothing_context():
            color_info = get_simul_meta(prob,
                                        repeats=options.num_jacs, tol=options.tolerance,
                                        show_jac=options.show_jac,
                                        include_sparsity=not options.no_sparsity,
                                        setup=True, run_model=True,
                                        stream=outfile)

        if sys.stdout.isatty():
            simul_coloring_summary(color_info, stream=sys.stdout)

        exit()
    return _simul_coloring


def _sparsity_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao sparsity' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', action='store', dest='outfile', help='output file (json format).')
    parser.add_argument('-n', action='store', dest='num_jacs', default=3, type=int,
                        help='number of times to repeat total derivative computation.')
    parser.add_argument('-t', action='store', dest='tolerance', default=1.e-15, type=float,
                        help='tolerance used to determine if a total jacobian entry is nonzero.')
    parser.add_argument('-j', '--jac', action='store_true', dest='show_jac',
                        help="Display a visualization of the final total jacobian used to "
                        "compute the sparsity.")


def _sparsity_cmd(options):
    """
    Return the post_setup hook function for 'openmdao sparsity'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    from openmdao.core.problem import Problem
    global _use_sparsity

    _use_sparsity = False

    def _sparsity(prob):
        if options.outfile is None:
            outfile = sys.stdout
        else:
            outfile = open(options.outfile, 'w')
        Problem._post_setup_func = None  # avoid recursive loop
        get_sparsity(prob, repeats=options.num_jacs, tol=options.tolerance, mode=prob._mode,
                     show_jac=options.show_jac, setup=True, run_model=True, stream=outfile)
        exit()
    return _sparsity
