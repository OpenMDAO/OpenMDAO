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
from contextlib import contextmanager

from six import iteritems
from six.moves import range

import numpy as np
from scipy.sparse.compressed import get_index_dtype
from scipy.sparse import coo_matrix

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.matrix import sparse_types
from openmdao.utils.array_utils import array_viz
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.mpi import MPI


CITATIONS = """
@article{Coleman+VermaSISC1998,
  author  = {Thomas F. Coleman and Arun Verma},
  title   = {The Efficient Computation of Sparse Jacobian Matrices Using Automatic Differentiation},
  journal = {SIAM Journal of Scientific Computing},
  year    = 1998,
  number  = 4,
  pages   = {1210-1233},
  month   = 7,
  volume  = 19
}
"""

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

    if ncols == 0:
        return

    # use max degree column as a starting point instead of just choosing a random column
    # since all have incidence degree of 0 when we start.
    start = degrees.argmax()
    yield start

    colored_degrees = np.zeros(degrees.size, dtype=get_index_dtype(maxval=degrees[start]))
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

    # zero out diagonal (column is not adjacent to itself)
    np.fill_diagonal(col_matrix, False)

    return col_matrix


def _Jc2col_matrix_direct(J, Jc):
    """
    Convert a partitioned jacobian sparsity matrix to a column adjacency matrix.

    This creates the column adjacency matrix used for direct jacobian determination
    as described in Coleman, T.F., Verma, A. (1998) The efficient Computation of Sparse Jacobian
    Matrices Using Automatic Differentiation. SIAM Journal on Scientific Computing, 19(4),
    1210-1233.

    Parameters
    ----------
    J : ndarray
        Boolean jacobian sparsity matrix.
    Jc : ndarray
        Boolean sparsity matrix of a partition of J.

    Returns
    -------
    ndarray
        Column adjacency matrix.
    """
    assert J.shape == Jc.shape

    nrows, ncols = J.shape

    col_matrix = np.zeros((ncols, ncols), dtype=bool)

    col_keep = _count_nonzeros(Jc, axis=0) > 0

    # mark col_matrix[col1, col2] as True when Jc[row, col1] is True OR Jc[row, col2] is True
    for row in range(nrows):
        nzro = np.nonzero(J[row] & col_keep)[0]
        for col1, col2 in combinations(nzro, 2):
            if Jc[row, col1] or Jc[row, col2]:
                col_matrix[col1, col2] = True
                col_matrix[col2, col1] = True

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
    return _get_full_disjoint_col_matrix_cols(_J2col_matrix(J))


def _get_full_disjoint_col_matrix_cols(col_matrix):
    """
    Find sets of disjoint columns in a column intersection matrix.

    Parameters
    ----------
    col_matrix : ndarray
        Column intersection matrix

    Returns
    -------
    list
        List of lists of disjoint columns
    """
    color_groups = []
    _, ncols = col_matrix.shape

    # -1 indicates that a column has not been colored
    colors = np.full(ncols, -1, dtype=get_index_dtype(maxval=ncols))

    for col in _order_by_ID(col_matrix):
        neighbor_colors = set(colors[col_matrix[col]])
        for color, grp in enumerate(color_groups):
            if color not in neighbor_colors:
                grp.append(col)
                colors[col] = color
                break
        else:
            colors[col] = len(color_groups)
            color_groups.append([col])

    return color_groups


def _color_partition(J, Jpart):
    """
    Compute a single directional fwd coloring using partition Jpart.

    This routine is used to compute a fwd coloring on Jc and a rev coloring on Jr.T.

    Parameters
    ----------
    J : ndarray
        Dense jacobian sparsity matrix
    Jpart : ndarray
        Partition of the jacobian sparsity matrix.

    Returns
    -------
    list
        List of color groups.  First group is uncolored.
    list
        List of nonzero rows for each column.
    """
    ncols = Jpart.shape[1]
    col_nonzeros = _count_nonzeros(Jpart, axis=0)
    row_nonzeros = _count_nonzeros(Jpart, axis=1)
    col_keep = col_nonzeros > 0
    row_keep = row_nonzeros > 0

    # use this to map indices back to the full J indices.
    idxmap = np.arange(ncols, dtype=int)[col_keep]

    intersection_mat = _Jc2col_matrix_direct(J, Jpart)
    intersection_mat = intersection_mat[col_keep]
    intersection_mat = intersection_mat[:, col_keep]

    col_groups = _get_full_disjoint_col_matrix_cols(intersection_mat)

    for i, group in enumerate(col_groups):
        col_groups[i] = sorted([idxmap[c] for c in group if col_keep[idxmap[c]]])
    col_groups = _split_groups(col_groups)

    col2row = [None] * ncols
    for col in idxmap:
        col2row[col] = [r for r in np.nonzero(Jpart[:, col])[0] if row_keep[r]]

    return [col_groups, col2row]


def MNCO_bidir(J):
    """
    Compute bidirectional coloring using Minimum Nonzero Count Order (MNCO).

    Based on the algorithm found in Coleman, T.F., Verma, A. (1998) The efficient Computation
    of Sparse Jacobian Matrices Using Automatic Differentiation. SIAM Journal on Scientific
    Computing, 19(4), 1210-1233.

    Parameters
    ----------
    J : ndarray
        Dense Jacobian sparsity matrix

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
    """
    nrows, ncols = J.shape

    M_col_nonzeros = _count_nonzeros(J, axis=0)
    M_row_nonzeros = _count_nonzeros(J, axis=1)

    M = coo_matrix(J)
    M_rows = M.row
    M_cols = M.col
    M = None  # we don't need sparse matrix data array, so clean up memory

    Jc_rows = [None] * nrows
    Jr_cols = [None] * ncols

    row_i = col_i = 0

    # partition J into Jc and Jr
    # We build Jc from bottom up and Jr from right to left.
    r = M_row_nonzeros.argmin()
    c = M_col_nonzeros.argmin()

    nnz_r = M_row_nonzeros[r]
    nnz_c = M_col_nonzeros[c]

    Jc_nz_max = 0   # max row nonzeros in Jc
    Jr_nz_max = 0   # max col nonzeros in Jr

    while M_rows.size + M_cols.size > 0:
        if Jr_nz_max + max(Jc_nz_max, nnz_r) < (Jc_nz_max + max(Jr_nz_max, nnz_c)):
            Jc_rows[r] = M_cols[M_rows == r]
            Jc_nz_max = max(nnz_r, Jc_nz_max)

            keep = M_rows != r
            M_rows = M_rows[keep]
            M_cols = M_cols[keep]

            M_row_nonzeros[r] = ncols + 1  # make sure we don't pick this one again
            M_col_nonzeros[Jc_rows[r]] -= 1

            r = M_row_nonzeros.argmin()
            nnz_r = M_row_nonzeros[r]

            row_i += 1
        else:
            Jr_cols[c] = M_rows[M_cols == c]
            Jr_nz_max = max(nnz_c, Jr_nz_max)

            keep = M_cols != c
            M_rows = M_rows[keep]
            M_cols = M_cols[keep]

            M_col_nonzeros[c] = nrows + 1  # make sure we don't pick this one again
            M_row_nonzeros[Jr_cols[c]] -= 1

            c = M_col_nonzeros.argmin()
            nnz_c = M_col_nonzeros[c]

            col_i += 1

    coloring = {'nrows': nrows, 'ncols': ncols}

    nnz_Jc = nnz_Jr = 0
    jac = np.zeros(J.shape, dtype=bool)

    if row_i > 0:
        Jc = jac
        # build Jc and do fwd coloring on it
        for i, cols in enumerate(Jc_rows):
            if cols is not None:
                Jc[i][cols] = True
                nnz_Jc += len(cols)

        coloring['fwd'] = _color_partition(J, Jc)
        jac[:] = False

    if col_i > 0:
        Jr = jac
        # build Jr and do rev coloring
        for i, rows in enumerate(Jr_cols):
            if rows is not None:
                Jr[rows, i] = True
                nnz_Jr += len(rows)

        coloring['rev'] = _color_partition(J.T, Jr.T)

    if np.count_nonzero(J) != nnz_Jc + nnz_Jr:
        raise RuntimeError("Nonzero mismatch for J vs. Jc and Jr")

    # _check_coloring(J, coloring)

    return coloring


def _tol_sweep(arr, tol=1e-15, orders=5):
    """
    Find best tolerance 'around' tol to choose nonzero values of arr.

    Sweeps over tolerances +- 'orders' orders of magnitude around tol and picks the most
    stable one (one corresponding to the most repeated number of nonzero entries).

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


@contextmanager
def _computing_coloring_context(top):
    """
    Context manager for computing total jac sparsity for simultaneous coloring.

    Parameters
    ----------
    top : System
        Top of the system hierarchy where coloring will be done.
    """
    for system in top.system_iter(recurse=True, include_self=True):
        if system.matrix_free:
            raise RuntimeError("%s: simultaneous coloring does not currently work with matrix free "
                               "components." % system.pathname)

        jac = system._assembled_jac
        if jac is None:
            jac = system._jacobian
        if jac is not None:
            jac._randomize = True

    yield

    for system in top.system_iter(recurse=True, include_self=True):
        jac = system._assembled_jac
        if jac is None:
            jac = system._jacobian
        if jac is not None:
            jac._randomize = False


def _get_bool_total_jac(prob, repeats=3, tol=1e-15, orders=5, setup=False, run_model=False):
    """
    Return a boolean version of the total jacobian.

    The jacobian is computed by calculating a total jacobian using _compute_totals 'repeats'
    times and adding the absolute values of those together, then dividing by 'repeats',
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
        prob.run_model(reset_iter_counts=False)

    with _computing_coloring_context(prob.model):
        start_time = time.time()
        fullJ = None
        for i in range(repeats):
            J = prob.driver._compute_totals(return_format='array')
            if fullJ is None:
                fullJ = np.abs(J)
            else:
                fullJ += np.abs(J)
        elapsed = time.time() - start_time

    fullJ /= repeats

    good_tol, nz_matches, n_tested, zero_entries = _tol_sweep(fullJ, tol, orders)

    print("\nUsing tolerance: %g" % good_tol)
    print("Most common number of zero entries (%d of %d) repeated %d times out of %d tolerances "
          "tested.\n" % (zero_entries, fullJ.size, nz_matches, n_tested))
    print("Full total jacobian was computed %d times, taking %f seconds." % (repeats, elapsed))
    print("Total jacobian shape:", fullJ.shape, "\n")

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
    for i, out in enumerate(sorted(sparsity)):
        out_dict = sparsity[out]
        stream.write('"%s": {\n' % out)
        last_dv_idx = len(out_dict) - 1
        for j, inp in enumerate(sorted(out_dict)):
            subjac = out_dict[inp]
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
            if isinstance(nonzeros, np.ndarray):
                # convert to list to make json serializable
                stream.write("   %s" % list(nonzeros))
            else:  # a full slice
                stream.write("   %s" % none)

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


def _json2coloring(coloring):
    """
    Convert all of the None entries in rowcol_map to full slices.

    Parameters
    ----------
    coloring : dict
        Dict of coloring metadata.

    Returns
    -------
    dict
        Dict of coloring metadata.
    """
    full_slice = slice(None)
    for mode in ('fwd', 'rev'):
        if mode in coloring:
            rcmap = coloring[mode][1]
            for i, entry in enumerate(rcmap):
                if entry is None:
                    rcmap[i] = full_slice

    return coloring


def get_tot_jac_sparsity(problem, mode='fwd', repeats=1, tol=1.e-15, show_jac=False,
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

    J = _get_bool_total_jac(problem, repeats=repeats, tol=tol, setup=setup, run_model=run_model)

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


def _total_solves(color_info, do_fwd=True, do_rev=True):
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
    do_fwd : bool
        If True, add fwd colors to total.
    do_rev : bool
        If True, add rev colors to total.

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
    if do_fwd and 'fwd' in color_info:
        row_lists, _ = color_info['fwd']
        total_solves += len(row_lists[0]) + len(row_lists) - 1
    if do_rev and 'rev' in color_info:
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

    Returns
    -------
    float
        Total size (minimum chosen based on which mode is better).
    float
        Total solves.
    """
    rev_size = color_info.get('nrows', -1)
    fwd_size = color_info.get('ncols', -1)

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

        if tot_size <= 0:
            pct = 0.
        else:
            pct = ((tot_size - tot_colors) / tot_size * 100)

    if tot_size < 0:
        tot_size = '?'

    return tot_size, tot_colors, fwd_solves, rev_solves, pct


def _split_groups(groups):
    uncolored = [grp[0] for grp in groups if len(grp) == 1]
    groups = [grp for grp in groups if len(grp) > 1]

    # the first lists entry corresponds to all uncolored columns (columns that are not
    # disjoint wrt any other columns).  The other entries are groups of columns that do not
    # share any nonzero row entries in common.
    clists = [uncolored]
    clists.extend(groups)

    return clists


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
    """
    bidirectional = mode == 'auto'
    rev = mode == 'rev'
    nrows, ncols = J.shape

    if bidirectional:
        return MNCO_bidir(J)

    if rev:
        J = J.T
    col_groups = _split_groups(_get_full_disjoint_cols(J))

    full_slice = slice(None)
    col2rows = [full_slice] * J.shape[1]  # will contain list of nonzero rows for each column
    for lst in col_groups:
        for col in lst:
            col2rows[col] = np.nonzero(J[:, col])[0]

    return {mode: [col_groups, col2rows], 'nrows': nrows, 'ncols': ncols}


def color_iterator(coloring, direction):
    """
    Given a list of column groupings and a mapping of nonzero rows per column, iterate over colors.

    Parameters
    ----------
    coloring : dict
        dict[direction] = (col_lists, row_maps)
            col_lists is a list of column lists, the first being a list of uncolored columns.
            row_maps is a list of nonzero rows for each column, or None for uncolored columns.
        Dict may contain either 'fwd' subdict, 'rev' subdict, or both.
    direction : str
        Indicates which coloring subdict ('fwd' or 'rev') to use.
    """
    col_lists = coloring[direction][0]
    nz_rows = coloring[direction][1]

    for i, col_chunk in enumerate(col_lists):
        if i == 0:
            # first chunk is a list of uncolored columns
            for c in col_chunk:
                yield [c], [nz_rows[c]]
        else:
            yield col_chunk, [nz_rows[c] for c in col_chunk]


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
        J = _get_bool_total_jac(problem, repeats=repeats, tol=tol, setup=setup,
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
            colored_array_viz(J, coloring, prob=problem, of=of, wrt=wrt, stream=s)

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

    nrows = color_info.get('nrows', -1)
    ncols = color_info.get('ncols', -1)

    if 'fwd' not in color_info and 'rev' not in color_info:
        tot_size = min(nrows, ncols)
        if tot_size < 0:
            tot_size = '?'
        stream.write("\nSimultaneous derivatives can't improve on the total number of solves "
                     "required (%s) for this configuration\n" % tot_size)
    else:
        tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(color_info)

        stream.write("\nFWD solves: %d   REV solves: %d" % (fwd_solves, rev_solves))
        stream.write("\n\nTotal colors vs. total size: %d vs %s  (%.1f%% improvement)\n" %
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
    if not problem.model._use_derivatives:
        simple_warning("Derivatives have been turned off. Skipping dynamic sparsity computation.")
        return

    driver._total_jac = None
    repeats = driver.options['dynamic_derivs_repeats']

    # save the sparsity.json file for later inspection
    with open("sparsity.json", "w") as f:
        sparsity = get_tot_jac_sparsity(problem, mode=problem._mode, repeats=repeats, stream=f)

    driver.set_total_jac_sparsity(sparsity)
    driver._setup_tot_jac_sparsity()


def dynamic_simul_coloring(driver, run_model=True, do_sparsity=False, show_jac=False):
    """
    Compute simultaneous deriv coloring during runtime.

    Parameters
    ----------
    driver : <Driver>
        The driver performing the optimization.
    run_model : bool
        If True, call run_model before computing coloring.
    do_sparsity : bool
        If True, setup the total jacobian sparsity (needed by pyOptSparseDriver).
    show_jac : bool
        If True, display a visualization of the colored jacobian.
    """
    problem = driver._problem
    if not problem.model._use_derivatives:
        simple_warning("Derivatives have been turned off. Skipping dynamic simul coloring.")
        return

    driver._total_jac = None

    # save the coloring.json file for later inspection
    with open("coloring.json", "w") as f:
        coloring = get_simul_meta(problem,
                                  repeats=driver.options['dynamic_derivs_repeats'],
                                  tol=1.e-15, include_sparsity=do_sparsity,
                                  setup=False, run_model=run_model, show_jac=show_jac, stream=f)
    driver.set_simul_deriv_color(coloring)
    driver._setup_simul_coloring()
    if do_sparsity:
        driver._setup_tot_jac_sparsity()

    simul_coloring_summary(coloring, stream=sys.stdout)


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
        if prob.model._use_derivatives:
            Problem._post_setup_func = None  # avoid recursive loop

            with profiling('coloring_profile.out') if options.profile else do_nothing_context():
                color_info = get_simul_meta(prob,
                                            repeats=options.num_jacs, tol=options.tolerance,
                                            show_jac=options.show_jac,
                                            include_sparsity=not options.no_sparsity,
                                            setup=False, run_model=True,
                                            stream=outfile)

            if sys.stdout.isatty():
                simul_coloring_summary(color_info, stream=sys.stdout)
        else:
            print("Derivatives are turned off.  Cannot compute simul coloring.")
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
        get_tot_jac_sparsity(prob, repeats=options.num_jacs, tol=options.tolerance, mode=prob._mode,
                             show_jac=options.show_jac, setup=True, run_model=True, stream=outfile)
        exit()
    return _sparsity


def _check_coloring(J, coloring):
    """
    Raise an exception if any problems are found with the coloring info.

    Parameters
    ----------
    J : ndarray
        Jacobian sparsity matrix.
    coloring : dict
        Metadata required for coloring.
    """
    # check for any overlapping nonzeros
    fwd_coloring = coloring.get('fwd')
    rev_coloring = coloring.get('rev')

    if fwd_coloring is not None:
        col_idx_groups = fwd_coloring[0]
        all_cols = set()
        for grp in col_idx_groups:
            all_cols.update(grp)

        col2row = fwd_coloring[1]
        fwd_nzs = []
        for col, rows in enumerate(col2row):
            if rows is not None:
                fwd_nzs.extend([(r, col) for r in rows])
            elif col in all_cols:  # full solve
                full_nzs = [(r, col) for r in np.nonzero(J[:, col])[0]]
                fwd_nzs.extend(full_nzs)

        fwd_nz_set = set(fwd_nzs)
        if len(fwd_nzs) != len(fwd_nz_set):
            raise RuntimeError("Duplicated nonzeros found in fwd coloring.")

        # find any nonzeros that are left out of any of our columns
        for c in all_cols:
            rnz = col2row[c]
            Jnz = np.nonzero(J[:, c])[0]
            missing = set(Jnz).difference(rnz)
            print("missing for col %d" % c, missing)

    if rev_coloring is not None:
        row_idx_groups = rev_coloring[0]
        all_rows = set()
        for grp in row_idx_groups:
            all_rows.update(grp)

        row2col = rev_coloring[1]
        rev_nzs = []
        for row, cols in enumerate(row2col):
            if cols is not None:
                rev_nzs.extend([(row, c) for c in cols])
            elif row in all_rows:  # full solve
                rev_nzs.extend([(row, c) for c in np.nonzero(J[row])[0]])

        rev_nz_set = set(rev_nzs)
        if len(rev_nzs) != len(rev_nz_set):
            raise RuntimeError("Duplicated nonzeros found in rev coloring.")

        # find any nonzeros that are left out of any of our columns
        for r in all_rows:
            cnz = row2col[r]
            Jnz = np.nonzero(J[r])[0]
            missing = set(Jnz).difference(cnz)
            print("missing for row %d" % r, missing)

    if fwd_coloring is not None and rev_coloring is not None:
        common = fwd_nz_set.intersection(rev_nzs)
        if common:
            raise RuntimeError("Coloring has overlapping nonzeros: %s" % list(common))

        computed_nzs = len(fwd_nz_set) + len(rev_nz_set)
        nzs = np.count_nonzero(J)
        if computed_nzs != nzs:
            raise RuntimeError("Colored nonzeros (%d) != nonzeros in J (%d)" % (computed_nzs, nzs))


def colored_array_viz(arr, coloring, prob=None, of=None, wrt=None, stream=sys.stdout):
    """
    Display the structure of a boolean array with coloring info for each nonzero value.

    Forward mode colored nonzeros are denoted by 'f', reverse mode nonzeros by 'r',
    overlapping nonzeros by 'O' and uncolored nonzeros by 'x'.  Zeros are denoted by '.'.

    If prob, of, and wrt are supplied, print the name of the response alongside
    each row and print the names of the design vars, aligned with each column, at
    the bottom.

    Parameters
    ----------
    arr : ndarray
        Array being visualized.
    coloring : dict
        Metadata needed for coloring.
    prob : Problem or None
        Problem object.
    of : list of str or None
        Names of response variables used in derivative calculation.
    wrt : list of str or None
        Names of design variables used in derivative calculation.
    stream : file-like
        Stream where output will be written.
    """
    if len(arr.shape) != 2:
        raise RuntimeError("array_viz only works for 2d arrays.")

    charr = np.full(arr.shape, '.', dtype=str)

    full_rows = np.arange(arr.shape[0], dtype=int)
    full_cols = np.arange(arr.shape[0], dtype=int)

    if 'fwd' in coloring:
        col2row = coloring['fwd'][1]
        for grp in coloring['fwd'][0]:
            for c in grp:
                rows = col2row[c]
                if rows is None:
                    rows = full_rows
                charr[rows, c] = 'f'

    if 'rev' in coloring:
        row2col = coloring['rev'][1]
        for grp in coloring['rev'][0]:
            for r in grp:
                cols = row2col[r]
                if cols is None:
                    cols = full_cols
                for c in cols:
                    if charr[r, c] == 'f':
                        # overlap!
                        charr[r, c] = 'O'
                    else:
                        charr[r, c] = 'r'

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c] and charr[r, c] == '.':
                charr[r, c] = 'x'  # mark nonzero as uncolored

    if prob is None or of is None or wrt is None:
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                stream.write(charr[r, c])
            stream.write(' %d\n' % r)
    else:
        row = 0
        for res in of:
            for r in range(row, row + prob.driver._responses[res]['size']):
                col = 0
                for dv in wrt:
                    for c in range(col, col + prob.driver._designvars[dv]['size']):
                        stream.write(charr[r, c])
                    col = c + 1
                stream.write(' %d  %s\n' % (r, res))
            row = r + 1

        start = 0
        for name in wrt:
            tab = ' ' * start
            stream.write('%s|%s\n' % (tab, name))
            start += prob.driver._designvars[name]['size']
