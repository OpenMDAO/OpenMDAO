"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import os
import sys
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
            count = np.zeros(arr.shape[1], dtype=int)
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


def _get_full_disjoint_cols(J):
    """
    Find sets of disjoint columns in J and their corresponding rows.

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

    # Start with col_matrix all True, meaning assume all columns are disjoint.
    # Note that col_matrix is symmetric.
    col_matrix = np.ones((ncols, ncols), dtype=bool)

    # mark col_matrix entries as False when nonzero row entries make them non-disjoint
    for row in range(nrows):
        nzro = np.nonzero(J[row])[0]
        for col in nzro:
            col_matrix[col, nzro] = False
            col_matrix[nzro, col] = False

    # count the number of pairwise disjoint columns in each column of col_matrix
    disjoint_counts = _count_nonzeros(col_matrix, axis=0)

    seen = set()
    colors = []

    # create a reusable rows vector for checking disjointness
    allrows = np.zeros(J.shape[0], dtype=bool)

    # loop over columns sorted in order of disjointness, smallest number of disjoint cols first
    for col in np.argsort(disjoint_counts):
        if col in seen:
            continue
        seen.add(col)
        allrows[:] = J[:, col]
        color = [col]
        colors.append(color)
        # col_matri[col, :] contains all columns that could possibly share the same color. Not all
        # of them generally will though since pairwise disjointness is not transitive.
        for other_col in np.nonzero(col_matrix[col, :])[0]:
            if other_col not in seen and not np.any(allrows & J[:, other_col]):
                seen.add(other_col)
                color.append(other_col)
                allrows |= J[:, other_col]

    return colors


def _get_bool_jac(prob, repeats=3, tol=1e-15, setup=False, run_model=False):
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

    fullJ = None
    for i in range(repeats):
        J = prob.driver._compute_totals(return_format='array', of=of, wrt=wrt)
        if fullJ is None:
            fullJ = np.abs(J)
        else:
            fullJ += np.abs(J)

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

    # normalize the full J by dividing by the max value
    fullJ /= np.max(fullJ)

    # sweep over tolerances +- 'orders' orders of magnitude from given tolerance and pick the most
    # stable one (one corresponding to the most repeated number of nonzero entries).
    orders = 5
    nzeros = defaultdict(list)
    itol = tol * 10.**orders
    smallest = tol / 10.**orders
    n_tested = 0
    while itol >= smallest:
        if itol < 1.:
            num_zero = fullJ[fullJ <= itol].size
            nzeros[num_zero].append(itol)
            n_tested += 1
        itol /= 10.

    # pick lowest tolerance corresponding to the most repeated number of 'zero' entries
    sorted_items = sorted(nzeros.items(), key=lambda x: len(x[1]), reverse=True)
    good_tol = sorted_items[0][1][-1]
    print("\nUsing tolerance: %g" % good_tol)
    print("Most common number of zero entries (%d of %d) repeated %d times out of %d tolerances "
          "tested.\n" % (sorted_items[0][0], fullJ.size, len(sorted_items[0][1]), n_tested))
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
        total_solves += len(row_lists[0]) + len(row_lists[1:])
    if 'rev' in color_info:
        col_lists, _ = color_info['rev']
        total_solves += len(col_lists[0]) + len(col_lists[1:])

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

    if tot_colors == 0:  # no coloring found
        tot_colors = tot_size = min([rev_size, fwd_size])
        colored_solves = opp_solves = 0
        dominant_mode = 'None'
        pct = 0.
    else:
        fwd_lists = color_info['fwd'][0] if 'fwd' in color_info else []
        rev_lists = color_info['rev'][0] if 'rev' in color_info else []

        dominant_mode = 'fwd' if len(fwd_lists) > len(rev_lists) else 'rev'

        if dominant_mode == 'fwd':
            tot_size = fwd_size
            colored_solves = len(fwd_lists[0]) + len(fwd_lists) - 1
            opp_solves = len(rev_lists[0])
        else:
            tot_size = rev_size
            colored_solves = len(rev_lists[0]) + len(rev_lists) - 1
            opp_solves = len(fwd_lists[0])

        pct = ((tot_size - tot_colors) / tot_size * 100)

    return tot_size, tot_colors, colored_solves, opp_solves, pct, dominant_mode


def _compute_coloring(J, mode, bidirectional, simul_coloring_excludes):
    """
    Compute the best coloring in a specified dominant direction.

    The f_coloring function determines the direction of the dominant coloring.

    Parameters
    ----------
    J : ndarray
        The boolean total jacobian.
    mode : str
        The dominant direction for solving for total derivatives.
    bidirectional : bool
        If True, compute a bidirectional coloring.
    simul_coloring_excludes : iter of int
        A collection of rows (fwd) or cols (rev) that are to be excluded from the coloring and
        solved in the opposite direction.

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
    rev = mode == 'rev'

    if rev:
        orig_J = J
        J = J.copy().T
    else:  # fwd
        orig_J = J
        J = J.copy()

    ###################################
    # Bidirectional coloring algorithm
    ###################################
    #
    # 1. Compute density of nonzero values for all rows and columns.
    # 2. Compute initial coloring on full jacobian.
    # 3. In a loop, zero out rows (fwd) or cols (rev) one at a time, most dense first, then
    #    recompute coloring and see if we get better.  If any are fully dense, zero out all of them
    #    at the same time since getting a coloring with dense rows (fwd) or cols (rev) isn't
    #    possible. Note that 'better' means the total solves in
    #    the current direction plus the number of dense solves in the opposite direction is less
    #    than our current best.
    #
    # Note that when we're done, the coloring for the chosen direction will contain a list of
    #     column or row lists, with the first entry containing the indices of the uncolored
    #     rows or cols for that direction, and the coloring for the opposite direction will
    #     specify only uncolored solves, so it will have only 1 entry in its list of lists.
    #
    # Wnen we solve for the total jacobian, we order the solves such that all of the colored ones
    # are done first, doing the dense ones last so that they'll overwrite any incorrect values
    # in the jacobian resulting from our earlier colored solves.

    coloring = {'J': J, 'rev': [[[]], []]}
    best_colors = J.shape[1]
    best_coloring = None
    tot_size = J.shape[1]

    if bidirectional:
        # get density of rows
        max_score = 1.0
        row_score = _count_nonzeros(J, axis=1) / J.shape[1]
        if simul_coloring_excludes is not None:
            max_score = np.max(row_score) + 1.0
            # make score highest for any explicitly excluded rows
            row_score[simul_coloring_excludes] = max_score

        full_dense = row_score[row_score == max_score]

        opp_solve_rows = np.argsort(row_score)[::-1]  # ordered highest score to lowest

        # we can use the max degree of a row to avoid wasting our time with certain iterations
        row_degree = _count_nonzeros(J, axis=1)

        # if we have any dense rows, zero them all out since we can't get a coloring with them.
        if full_dense.size > 0:
            num_opp_solves = full_dense.size
            row_degree[opp_solve_rows[:num_opp_solves]] = 0
        else:  # coloring is possible.  Do coloring for unaltered J to get initial best
            num_opp_solves = 0
    else:
        num_opp_solves = 0

    # bail if trying any additional opp_solves can't do better than our curent best
    while num_opp_solves + 1 < best_colors:

        if num_opp_solves > 0:
            J[opp_solve_rows[:num_opp_solves], :] = False  # zero out another skipped row
            row_degree[opp_solve_rows[num_opp_solves - 1]] = 0

            # If each of our rows is a node in a graph, we have to have at least as
            # many colors as the degree of that node, so overall we can't do better
            # than the max degree of all of our row nodes.  If that plus the number
            # of opposite direction solves is no better than our best coloring, then
            # don't bother running the algorthm on this version of J.
            if np.max(row_degree) + num_opp_solves >= best_colors:
                num_opp_solves += 1
                continue

        full_disjoint = _get_full_disjoint_cols(J)
        tot_colors = len(full_disjoint) + num_opp_solves

        if tot_colors < best_colors:
            best_colors = tot_colors

            uncolored_cols = [clist[0] for clist in full_disjoint if len(clist) == 1]
            full_disjoint = [clist for clist in full_disjoint if len(clist) > 1]

            # the first lists entry corresponds to all uncolored columns (columns that are not
            # disjoint wrt any other columns).  The other entries are groups of columns that do not
            # share any nonzero row entries in common.
            lists = [uncolored_cols]
            lists.extend(full_disjoint)

            rowcol_map = [None] * J.shape[1]  # will contain list of nonzero rows for each column
            for clist in full_disjoint:
                for col in clist:
                    # ndarrays are converted to lists to be json serializable
                    rowcol_map[col] = list(np.nonzero(J[:, col])[0])

            best_coloring = {
                'fwd': [lists, rowcol_map],
                'rev': [[list(opp_solve_rows[:num_opp_solves])], []],
                'J': orig_J
            }

        if not bidirectional:
            break

        num_opp_solves += 1  # add another row to solve in the opposite direction

    if best_coloring is None:
        best_coloring = {}
    else:
        if not best_coloring['rev'][0]:
            del best_coloring['rev']

        if rev:
            best_color = {'rev': best_coloring['fwd']}
            if 'rev' in best_coloring:
                best_color['fwd'] = best_coloring['rev']
            best_coloring = best_color

    best_coloring['J'] = orig_J

    return best_coloring


def _get_simul_excludes(problem):
    """
    Collect simul_coloring_excludes info from design vars or responses.

    Parameters
    ----------
    problem : Problem
        The Problem being run.

    Returns
    -------
    list of int
        List of jacobian row/col indices to exclude from the coloring (and solve in the
        opposite direction).
    """
    offset = 0
    simul_coloring_excludes = []
    if problem._mode == 'rev':
        wrt = list(problem.driver._designvars)
        desvars = problem.driver._designvars
        for dv in wrt:
            excl = desvars[dv]['simul_coloring_excludes']
            size = desvars[dv]['size']
            if excl:
                if isinstance(excl, bool):
                    simul_coloring_excludes.extend(np.arange(size) + offset)
                else:
                    simul_coloring_excludes.extend(np.array(excl) + offset)
            offset += size

    else:  # fwd
        of = problem.driver._get_ordered_nl_responses()
        resps = problem.driver._responses
        for resp in of:
            excl = resps[resp]['simul_coloring_excludes']
            size = resps[resp]['size']
            if excl:
                if isinstance(excl, bool):
                    simul_coloring_excludes.extend(np.arange(size) + offset)
                else:
                    simul_coloring_excludes.extend(np.array(excl) + offset)
            offset += size

    if not simul_coloring_excludes:
        return None

    return simul_coloring_excludes


def get_simul_meta(problem, mode=None, repeats=1, tol=1.e-15, show_jac=False,
                   include_sparsity=True, setup=False, run_model=False, bool_jac=None,
                   bidirectional=True, simul_coloring_excludes=None, stream=sys.stdout):
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
    bidirectional : bool
        If True, compute a bidirectional coloring.
    simul_coloring_excludes : iter of int
        A collection of rows (fwd) or cols (rev) that are to be excluded from the coloring and
        solved in the opposite direction. Used only if problem is None. Otherwise, problem
        driver will supply exclude information gathered from design vars or responses.
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
    """
    if problem is not None:
        J = _get_bool_jac(problem, repeats=repeats, tol=tol, setup=setup,
                          run_model=run_model)
    elif bool_jac is not None:
        J = bool_jac
    else:
        raise RuntimeError("You must supply either problem or bool_jac to get_simul_meta().")

    if mode is None:
        mode = problem._mode

    if problem is not None:
        simul_coloring_excludes = _get_simul_excludes(problem)

    coloring = _compute_coloring(J, mode, bidirectional, simul_coloring_excludes)

    modes = [m for m in ('fwd', 'rev') if m in coloring]

    sparsity = None
    if problem is not None:
        driver = problem.driver
    else:
        driver = None

    if driver is not None:
        if include_sparsity or (show_jac and stream is not None):
            of = driver._get_ordered_nl_responses()
            wrt = list(driver._designvars)

        if include_sparsity:
            sparsity = _sparsity_from_jac(J, of, wrt, driver)
            coloring['sparsity'] = sparsity

        driver._total_jac = None

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
    if 'fwd' not in color_info and 'rev' not in color_info:
        tot_size = min(color_info['J'].shape)
        stream.write("Simultaneous derivatives can't improve on the total number of solves "
                     "required (%d) for this configuration\n" % tot_size)
    else:
        tot_size, tot_colors, colored_solves, opp_solves, pct, mode = _solves_info(color_info)

        stream.write("\nColored solves in %s mode: %d   opposite solves: %d" %
                     (mode, colored_solves, opp_solves))
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
    driver._setup_simul_coloring(mode=problem._mode)
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
