"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import sys
import warnings
from collections import OrderedDict, defaultdict
from itertools import combinations

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


def _get_full_disjoint_cols(J, start, end):
    """
    Find sets of disjoint columns between start and end in J and their corresponding rows.

    Parameters
    ----------
    J : ndarray
        The total jacobian.
    start : int
        The starting column.
    end : int
        The ending column.

    Returns
    -------
    (list, dict)
        List of lists of disjoint columns and lists of nonzero rows by column.
    """
    # skip desvars of size 1 since simul derivs will give no improvement
    if (end - start) == 0:
        return {}, {}

    disjoints = defaultdict(set)
    rows = [None] * J.shape[1]  # will contain list of nonzero rows for each column
    for c1, c2 in combinations(range(start, end + 1), 2):  # loop over column pairs
        # 'and' two columns together. If we get all False, then columns have disjoint row sets
        if not np.any(J[:, c1] & J[:, c2]):
            disjoints[c1].add(c2)
            disjoints[c2].add(c1)
            # ndarrays are converted to lists to be json serializable
            if rows[c1] is None:
                rows[c1] = [int(i) for i in np.nonzero(J[:, c1])[0]]
            if rows[c2] is None:
                rows[c2] = [int(i) for i in np.nonzero(J[:, c2])[0]]

    full_disjoint = []
    seen = set()
    allrows = {}

    # sort largest to smallest disjoint column sets
    for col, colset in sorted(disjoints.items(), key=lambda x: len(x[1]), reverse=True):
        if col in seen:
            continue
        seen.add(col)
        allrows[col] = J[:, col].copy()
        full = [col]
        for other_col in colset:
            if other_col not in seen and not np.any(allrows[col] & J[:, other_col]):
                seen.add(other_col)
                full.append(other_col)
                allrows[col] |= J[:, other_col]

        if len(full) > 1:
            full_disjoint.append(sorted(full))
        else:
            rows[col] = None

    return sorted(full_disjoint, key=lambda x: len(x)), rows


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
    mode = prob._mode

    # TODO: should always automatically choose mode based on smallest number of rows or cols
    #       in the total jacobian (minus linear constraints)
    if setup:
        prob.setup(mode=mode)

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

    stream.write("{\n")
    for m, mode in enumerate(modes):
        name = 'column' if mode == 'fwd' else 'row'
        lists, nonzero_entries = color_info[mode]
        sparsity = color_info.get('sparsity')

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

        stream.write("]")
    stream.write("\n]")

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
    row_lists = col_lists = []
    total_solves = 0

    # lists[0] are the non-colored columns or rows, which are solved individually so
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


def _solves_info(color_info, dominant_mode):
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
    dominant_mode : str
        Derivative direction where coloring is performed.  Part of the solution
        (dense row/col solves only), may be done in the opposite direction.

    Returns
    -------
    float
        Total size (minimum chosed based on which mode is better).
    float
        Total solves.
    str
        Mode chosed based on lowest number of solves (without coloring).
    """
    rev_size, fwd_size = color_info['J'].shape
    tot_colors = _total_solves(color_info)

    if dominant_mode == 'fwd':
        tot_size = fwd_size
    else:
        tot_size = rev_size

    if tot_colors == 0:  # no coloring found
        tot_colors = tot_size

    return tot_size, tot_colors


def _compute_one_directional_coloring(J, mode, bidirectional):
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

    rev_rows = []
    coloring = {'J': J, 'rev': [[rev_rows], []]}
    best_colors = 99999999
    best_coloring = None

    if bidirectional:
        # get density of rows
        density = np.count_nonzero(J, axis=1) / J.shape[0]
        most_dense = np.argsort(density)[::-1]  # ordered most dense to least

        # if we have any dense rows, zero them out since we can't get a coloring with them.
        full_dense = density[density == 1.0]
        if full_dense.size > 0:
            skip_count = full_dense.size
        else:  # coloring is possible.  Do coloring for unaltered J to get initial best
            skip_count = 0
    else:
        skip_count = 0

    while skip_count < J.shape[0] / 2:
        if skip_count > 0:
            J[most_dense[skip_count - 1], :] = False  # zero out another skipped row
            rev_rows.append(most_dense[skip_count - 1])

        full_disjoint, rowcol_map = _get_full_disjoint_cols(J, 0, J.shape[1] - 1)
        uncolored_cols = [i for i, r in enumerate(rowcol_map) if r is None]

        # the first col_list entry corresponds to all uncolored columns (columns that are not
        # disjoint wrt any other columns).  The other entries are groups of columns that do not
        # share any nonzero row entries in common.
        lists = [uncolored_cols]
        lists.extend(full_disjoint)

        coloring['fwd'] = [lists, rowcol_map]

        tot_size, tot_colors = _solves_info(coloring, mode)

        # if we got worse, bail out
        if best_colors < tot_size and tot_colors >= best_colors:
            break

        if tot_colors < best_colors:
            best_colors = tot_colors

        best_coloring = {'fwd': coloring['fwd'], 'rev': [[rev_rows.copy()], []], 'J': orig_J}

        if not bidirectional:
            break

        skip_count += 1  # add another row to skip

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


def get_simul_meta(problem, repeats=1, tol=1.e-15, show_jac=False,
                   include_sparsity=True, setup=False, run_model=False, bool_jac=None,
                   bidirectional=True, stream=sys.stdout):
    """
    Compute simultaneous derivative colorings for the given problem.

    Parameters
    ----------
    problem : Problem or None
        The Problem being analyzed.
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

    if problem is not None:
        coloring = _compute_one_directional_coloring(J, problem._mode, bidirectional)
    else:
        best_fwd_coloring = _compute_one_directional_coloring(J, 'fwd', bidirectional)
        _, best_fwd_colors = _solves_info(best_fwd_coloring, 'fwd')
        best_rev_coloring = _compute_one_directional_coloring(J, 'rev', bidirectional)
        _, best_rev_colors = _solves_info(best_rev_coloring, 'rev')

        if best_fwd_colors <= best_rev_colors:
            coloring = best_fwd_coloring
        else:
            coloring = best_rev_coloring

        if 'rev' in best_fwd_coloring:
            fwd_opp = len(best_fwd_coloring['rev'][0][0])
        else:
            fwd_opp = 0
        if 'fwd' in best_rev_coloring:
            rev_opp = len(best_rev_coloring['fwd'][0][0])
        else:
            rev_opp = 0

        print("Best fwd:", best_fwd_colors, "fwd solves:", best_fwd_colors - fwd_opp,
              "opposite solves:", fwd_opp)
        print("Best rev:", best_rev_colors, "rev solves:", best_rev_colors - rev_opp,
              "opposite solves:", rev_opp)

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

    modes = [m for m in ('fwd', 'rev') if m in coloring]

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


def simul_coloring_summary(color_info, mode, stream=sys.stdout):
    """
    Print a summary of simultaneous coloring info for the given problem and coloring metadata.

    Parameters
    ----------
    color_info : dict
        Coloring metadata.
    mode : str
        Dominant derivative solution direction.
    stream : file-like
        Where the output will go.
    """
    tot_size, tot_colors = _solves_info(color_info, mode)

    if tot_size <= tot_colors:
        stream.write("Simultaneous derivatives can't improve on the total number of solves "
                     "required (%d) for %s mode in this configuration\n" % (tot_size, mode))
    else:
        stream.write("\nTotal colors vs. total size (%s mode): %d vs %d  (%.1f%% improvement)\n" %
                     (mode, tot_colors, tot_size, ((tot_size - tot_colors) / tot_size * 100)))


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
    parser.add_argument('-m', '--mode', action='store', dest='mode', default='fwd', type=str,
                        help='Direction of computation for derivatives.')
    parser.add_argument('-j', '--jac', action='store_true', dest='show_jac',
                        help="Display a visualization of the final total jacobian used to "
                        "compute the coloring.")
    parser.add_argument('--no-sparsity', action='store_true', dest='no_sparsity',
                        help="Exclude the sparsity structure from the coloring data structure.")


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
    global _use_sparsity

    _use_sparsity = False

    def _simul_coloring(prob):
        if options.outfile is None:
            outfile = sys.stdout
        else:
            outfile = open(options.outfile, 'w')
        Problem._post_setup_func = None  # avoid recursive loop
        color_info = get_simul_meta(prob,
                                    mode=options.mode,
                                    repeats=options.num_jacs, tol=options.tolerance,
                                    show_jac=options.show_jac,
                                    include_sparsity=not options.no_sparsity,
                                    setup=True, run_model=True,
                                    stream=outfile)
        if sys.stdout.isatty():
            simul_coloring_summary(color_info, options.mode, stream=sys.stdout)

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
