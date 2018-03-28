"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import os
import sys
import json

from collections import OrderedDict, defaultdict
from itertools import combinations, chain
from numbers import Integral

from six import iteritems, itervalues
from six.moves import range

import numpy as np
from numpy.random import rand

from openmdao.jacobians.jacobian import Jacobian
from openmdao.jacobians.assembled_jacobian import AssembledJacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.matrix import sparse_types
from openmdao.utils.array_utils import array_viz

# If this is True, then IF simul coloring is specified, use it.  If False, don't use it regardless.
# The command line simul_coloring command makes this False when generating a new coloring.
_use_simul_coloring = True


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
            info, shape = jac._subjacs_info[key]
            rows = info['rows']
        else:
            rows = None

        if rows is not None:  # list form
            subjac = rand(rows.size) + 1.0
        elif isinstance(subjac, sparse_types):  # sparse
            subjac = subjac.copy()
            subjac.data = rand(subjac.data.size) + 1.0
        else:   # dense
            subjac = rand(*(subjac.shape)) + 1.0

        self._orig_set_abs(key, subjac)


def _get_full_disjoint(J, start, end):
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
        List of lists of disjoint columns and dict of nonzero rows keyed to column.
    """
    # skip desvars of size 1 since simul derivs will give no improvement
    if (end - start) == 0:
        return {}, {}

    disjoints = defaultdict(set)
    rows = {}
    for c1, c2 in combinations(range(start, end + 1), 2):  # loop over column pairs
        # 'and' two columns together. If we get all False, then columns have disjoint row sets
        if not np.any(J[:, c1] & J[:, c2]):
            disjoints[c1].add(c2)
            disjoints[c2].add(c1)
            # ndarrays are converted to lists to be json serializable
            if c1 not in rows:
                rows[c1] = [int(i) for i in np.nonzero(J[:, c1])[0]]
            if c2 not in rows:
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

        if len(full) == 1:
            del rows[col]
        else:
            full_disjoint.append(sorted(full))

    return sorted(full_disjoint, key=lambda x: len(x)), rows


def _get_bool_jac(prob, mode='fwd', repeats=3, tol=1e-15, byvar=True):
    """
    Return a boolean version of the total jacobian.

    The jacobian is computed by calculating a total jacobian using _compute_totals 'repeats'
    times and adding the absolute values of those together, then dividing by the max value,
    then converting to a boolean array, specifying all entries below 'tol' as False and all
    others as True.  Prior to calling _compute_totals, all of the partial jacobians in the
    model are modified so that when any of their subjacobians are assigned a value, that
    value is populated with positive random numbers in the range [1.0, 2.0).

    Parameters
    ----------
    prob : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
    repeats : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance on values in jacobian.  Anything smaller in magnitude will be
        set to 0.0.
    byvar : bool
        If True, compute coloring by variable. Otherwise compute global coloring.

    Returns
    -------
    ndarray
        A boolean composite of 'repeats' total jacobians.
    """
    # TODO: fix this to work in rev mode as well
    assert mode == 'fwd', "Only fwd mode is supported."

    # clear out any old simul coloring info
    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    prob.setup(mode=mode)

    seen = set()
    for system in prob.model.system_iter(recurse=True, include_self=True):
        if system._jacobian not in seen:
            # replace jacobian set_abs with one that replaces all subjacs with random numbers
            system._jacobian._set_abs = _SubjacRandomizer(system._jacobian, tol)
            seen.add(system._jacobian)

    prob.run_model()

    wrt = list(prob.driver._designvars)

    # get responses in order used by the driver
    of = prob.driver.get_ordered_nl_responses()

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

    # normalize the full J
    fullJ /= np.max(fullJ)

    sz_zeros = fullJ[fullJ == 0.0].size
    print("smallest nonzero:", np.min(fullJ[fullJ > 0.0]))
    itol = tol * 1000.
    print("%d zero values out of %d (%5.2f%%)" % (sz_zeros, fullJ.size, sz_zeros/fullJ.size*100.))
    while itol > tol / 1000.:
        if itol < 1.:
            print("%d entries < tol at tol of %g" % (fullJ[fullJ < itol].size, itol))
        itol /= 10

    boolJ = np.zeros(fullJ.shape, dtype=bool)
    boolJ[fullJ > tol] = True

    return boolJ


def _compute_ranges(names, vois):
    """
    Get a list of varible ranges with one entry per row or column in the jacobian.

    Parameters
    ----------
    names : iter of str
        Names of vois.
    vois : dict
        Metadata of vois.

    Returns
    -------
    list
        List of size total_voi_size containing tuples of the form (start, end, name).
    """
    ranges = []
    start = 0
    end = -1
    for name in names:
        end += vois[name]['size']
        tup = (start, end, name)
        ranges.extend([tup] * (end - start + 1))
        start = end + 1

    return ranges


def _find_disjoint(prob, mode='fwd', repeats=1, tol=1e-15):
    """
    Find sets of disjoint columns by variable in the total jac and their corresponding rows.

    Parameters
    ----------
    prob : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
    repeats : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance on values in jacobian.  Anything smaller in magnitude will be
        set to 0.0.

    Returns
    -------
    tuple
        Tuple of the form (total_dv_offsets, total_res_offsets, final_jac)
    """
    # TODO: fix this to work in rev mode as well
    assert mode == 'fwd', "Only fwd mode is supported."

    J = _get_bool_jac(prob, mode=mode, repeats=repeats, tol=tol)

    # find column and row ranges (inclusive) for dvs and responses respectively
    dv_offsets = _compute_ranges(wrt, prob.driver._designvars)
    res_offsets = _compute_ranges(of, prob.driver._responses)

    total_dv_offsets = OrderedDict()
    total_res_offsets = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[], []])))

    # loop over each desvar and find disjoint column sets for all columns of that desvar
    for start, end, dv in dv_offsets:
        full_disjoint, rows = _get_full_disjoint(J, start, end)

        total_dv_offsets[dv] = tot_dv = OrderedDict()

        for color, cols in enumerate(full_disjoint):
            tot_dv[color] = tot_dv_columns = []
            for c in sorted(cols):
                dvoffset = c - start
                tot_dv_columns.append(dvoffset)
                for crow in rows[c]:
                    startcol, endcol, res = res_offsets[crow]
                    resoffset = crow - startcol
                    dct = total_res_offsets[res][dv][color]
                    # need to convert these to int to avoid error during JSON serialization
                    dct[0].append(int(resoffset))
                    dct[1].append(int(dvoffset))

    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    return total_dv_offsets, total_res_offsets, J


def _find_global_disjoint(prob, mode='fwd', repeats=1, tol=1e-15):
    """
    Find sets of disjoint columns in the total jac and their corresponding rows.

    Parameters
    ----------
    prob : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
    repeats : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance on values in jacobian.  Anything smaller in magnitude will be
        set to 0.0.

    Returns
    -------
    tuple
        Tuple of the form (disjoint_col_sets, rows_per_col)
    """
    # TODO: fix this to work in rev mode as well
    assert mode == 'fwd', "Only fwd mode is supported."

    J = _get_bool_jac(prob, mode=mode, repeats=repeats, tol=tol)

    full_disjoint, rows = _get_full_disjoint(J, 0, J.shape[1] - 1)

    for color, cols in enumerate(full_disjoint):
        print("color:", color, "columns:", len(cols))

    return full_disjoint, rows, J


def get_simul_meta(problem, mode='fwd', repeats=1, tol=1.e-15, show_jac=False,
                   include_sparsity=True, stream=sys.stdout):
    """
    Compute simultaneous derivative colorings for the given problem.

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
        If True, display a visualiation of the final total jacobian used to compute the coloring.
    include_sparsity : bool
        If True, include the sparsity structure of the total jacobian mapped to design vars
        and responses.  (This info is used by pyOptSparseDriver).
    stream : file-like or None
        Stream where output coloring info will be written.

    Returns
    -------
    tuple of the form (simul_colorings, simul_maps)
        Where simul_colorings is a dict of the form {dvname1: coloring_array, ...} and
        simul_maps is a dict of the form
        {resp_name: {dvname: {color: (row_idxs, col_idxs), ...}, ...}, ...}
    """
    driver = problem.driver

    full_disjoint, rows, J = _find_global_disjoint(problem, mode=mode, repeats=repeats, tol=tol)
    allcols = set(range(J.shape[1]))
    single_cols = list(allcols.difference(rows))

    # the first col_list entry corresponds to all single columns (columns that are not disjoint
    # wrt any other columns).  The other entries are groups of columns that are disjoint wrt
    # each other.
    col_lists = [single_cols]
    col_lists.extend(full_disjoint)

    rows = OrderedDict(sorted(rows.items(), key=lambda x: x[0]))

    sparsity = None
    if include_sparsity or (show_jac and stream is not None):
        of = driver.get_ordered_nl_responses()
        wrt = list(driver._designvars)

    if include_sparsity:
        sparsity = OrderedDict()
        row_start = row_end = 0
        for res in of:
            sparsity[res] = OrderedDict()
            res_size = int(driver._responses[res]['size'])
            row_end += res_size
            col_start = col_end = 0
            for dv in wrt:
                dv_size = int(driver._designvars[dv]['size'])
                col_end += dv_size

                # save sparsity structure as  (rows, cols, shape)
                irows, icols = np.nonzero(J[row_start:row_end, col_start:col_end])
                # convert to make JSON serializable
                irows = [int(i) for i in irows]
                icols = [int(i) for i in icols]
                sparsity[res][dv] = (irows, icols, (res_size, dv_size))

                col_start = col_end

            row_start = row_end

    if stream is not None:
        if stream.isatty():
            stream.write("\n([\n")
            for n, coloring in enumerate(col_lists):
                stream.write("   %s,\n" % coloring)
            stream.write("],")

            stream.write("\n{\n")
            for col, row_list in iteritems(rows):
                stream.write("   %s: %s,\n" % (col, list(row_list)))
            stream.write("}")

            if sparsity:
                import pprint
                stream.write(",\n")
                pprint.pprint(sparsity)

            stream.write(")\n")

        else:  # output json format to a file
            s = json.dumps((col_lists, rows, sparsity))

            # do a little pretty printing since the built-in json pretty printing stretches
            # the output vertically WAY too much.
            s = s.replace(',"', ',\n"')
            s = s.replace(', "', ',\n"')
            s = s.replace('{"', '{\n"')
            s = s.replace(', {', ',\n{')
            s = s.replace(']}', ']\n}')
            s = s.replace('{}', '{\n}')
            s = s.replace('}}', '}\n}')
            s = s.replace('[{', '[\n{')
            s = s.replace(' {', '\n{')

            lines = []
            indent = 0
            for line in s.split('\n'):
                start = line[0] if len(line) > 0 else ''
                if start in ('{', '['):
                    tab = ' ' * indent
                    indent += 3
                elif start in ('}', ']'):
                    indent -= 3
                    tab = ' ' * indent
                else:
                    tab = ' ' * indent

                lines.append("%s%s" % (tab, line))

            stream.write('\n'.join(lines))
            stream.write("\n")

        if show_jac and stream is not None:
            stream.write("\n\n")
            array_viz(J, problem, of, wrt, stream)

    return col_lists, rows, sparsity


def simul_coloring_summary(problem, color_info, stream=sys.stdout):
    """
    Print a summary of simultaneous coloring info for the given problem and coloring metadata.

    Parameters
    ----------
    problem : Problem
        The Problem being analyzed.
    color_info : tuple of (column_lists, row_map)
        Coloring metadata.
    stream : file-like
        Where the output will go.
    """
    column_lists, row_map, sparsity = color_info

    desvars = problem.driver._designvars

    tot_colors = len(column_lists[0]) + len(column_lists) - 1
    tot_size = 0

    if problem._mode == 'fwd':
        for dv in desvars:
            tot_size += desvars[dv]['size']
    else:  # rev
        raise RuntimeError("rev mode currently not supported for simultaneous derivs.")

    if sparsity is not None:
        stream.write("Sparsity structure has been computed for all response/design_var "
                     "sub-jacobians.")
    if tot_size == tot_colors:
        stream.write("No simultaneous derivative solves are possible in this configuration.\n")
    else:
        stream.write("\nTotal colors vs. total size: %d vs %d  (%.1f%% improvement)\n" %
                     (tot_colors, tot_size, ((tot_size - tot_colors) / tot_size * 100)))


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
    parser.add_argument('-n', action='store', dest='num_jacs', default=1, type=int,
                        help='number of times to repeat total derivative computation')
    parser.add_argument('-t', action='store', dest='tolerance', default=1.e-30, type=float,
                        help='tolerance used to determine if a total jacobian entry is nonzero')
    parser.add_argument('-j', '--jac', action='store_true', dest='show_jac',
                        help="Display a visualization of the final total jacobian used to "
                        "compute the coloring.")
    parser.add_argument('-s', '--sparsity', action='store_true', dest='include_sparsity',
                        help="Include the sparsity structure in the coloring data structure.")


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
    global _use_simul_coloring

    _use_simul_coloring = False

    def _simul_coloring(prob):
        if options.outfile is None:
            outfile = sys.stdout
        else:
            outfile = open(options.outfile, 'w')
        Problem._post_setup_func = None  # avoid recursive loop
        color_info = get_simul_meta(prob, repeats=options.num_jacs, tol=options.tolerance,
                                    show_jac=options.show_jac,
                                    include_sparsity=options.include_sparsity,
                                    stream=outfile)
        if sys.stdout.isatty():
            simul_coloring_summary(prob, color_info, stream=sys.stdout)

        exit()
    return _simul_coloring
