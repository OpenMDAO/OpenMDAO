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
    (OrderedDict, dict)
        Dict of disjoint columns keyed to column and dict of nonzero rows keyed to column.
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

    full_disjoint = OrderedDict()
    seen = set()
    allrows = {}

    # sort largest to smallest disjoint column sets
    for col, colset in sorted(disjoints.items(), key=lambda x: len(x[1]), reverse=True):
        if col in seen:
            continue
        seen.add(col)
        allrows[col] = J[:, col].copy()
        full_disjoint[col] = [col]
        for other_col in colset:
            if other_col not in seen and not np.any(allrows[col] & J[:, other_col]):
                seen.add(other_col)
                full_disjoint[col].append(other_col)
                allrows[col] |= J[:, other_col]

        if len(full_disjoint[col]) == 1:
            del full_disjoint[col]
            del rows[col]

    return full_disjoint, rows


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

    desvars = prob.driver._designvars
    responses = prob.driver._responses

    wrt = list(desvars)

    # remove linear constraints from consideration
    of = list(prob.driver._objs)
    of.extend(n for n, m in iteritems(prob.driver._cons) if not ('linear' in m and m['linear']))

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

        for color, cols in enumerate(itervalues(full_disjoint)):
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

    for color, cols in enumerate(itervalues(full_disjoint)):
        print("color", color, "cols", len(cols))

    return full_disjoint, rows, J


def get_simul_meta_old(problem, mode='fwd', repeats=1, tol=1.e-15, show_jac=False,
                       stream=sys.stdout):
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

    dv_idxs, res_idxs, J = _find_disjoint(problem, mode=mode, repeats=repeats, tol=tol)
    all_colors = set()

    simul_colorings = {}
    simul_maps = {}

    for dv in dv_idxs:
        # negative colors will be iterated over individually, so start by filling the coloring array
        # with -1.  We then replace specific entries with positive colors which will be iterated
        # over as a group.
        coloring = np.full(driver._designvars[dv]['size'], -1, dtype=int)

        for color in dv_idxs[dv]:
            coloring[np.array(dv_idxs[dv][color], dtype=int)] = color
            all_colors.add(color)

        if np.any(coloring != -1):
            # need int conversion to avoid JSON serialization error
            simul_colorings[dv] = [int(c) for c in coloring]

    simul_colorings = OrderedDict(sorted(simul_colorings.items()))

    for res in res_idxs:
        simul_map = {}
        for dv in res_idxs[res]:
            simul_map[dv] = {c: v for c, v in iteritems(res_idxs[res][dv])
                             if c in all_colors}
            if not simul_map[dv]:
                del simul_map[dv]

        if simul_map:
            simul_maps[res] = OrderedDict(sorted(simul_map.items()))

    simul_maps = OrderedDict(sorted(simul_maps.items()))

    if stream is not None:
        if stream.isatty():
            stream.write("\n({\n")
            for n, coloring in iteritems(simul_colorings):
                stream.write("   '%s': %s,\n" % (n, coloring))
            stream.write("},")

            stream.write("\n{\n")
            for res, dvdict in iteritems(simul_maps):
                stream.write("   '%s': {\n" % res)
                for dv, coldict in iteritems(dvdict):
                    stream.write("      '%s': {\n" % dv)
                    for color, idxs in iteritems(coldict):
                        stream.write("         %s: %s,\n" % (color, idxs))
                    stream.write("      },\n")
                stream.write("   },\n")
            stream.write("})")
        else:  # output json format to a file
            s = json.dumps((simul_colorings, simul_maps))

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
        of = list(driver._objs)
        of.extend([c for c, meta in iteritems(driver._cons)
                   if not ('linear' in meta and meta['linear'])])
        wrt = list(driver._designvars)

        stream.write("\n\n")
        array_viz(J, problem, of, wrt, stream)

    return simul_colorings, simul_maps


def get_simul_meta(problem, mode='fwd', repeats=1, tol=1.e-15, show_jac=False, stream=sys.stdout):
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
    col_lists.extend(full_disjoint.values())

    rows = OrderedDict(sorted(rows.items(), key=lambda x: x[0]))

    if stream is not None:
        if stream.isatty():
            stream.write("\n([\n")
            for n, coloring in enumerate(col_lists):
                stream.write("   %s,\n" % coloring)
            stream.write("],")

            stream.write("\n{\n")
            for col, row_list in iteritems(rows):
                stream.write("   %s: %s,\n" % (col, list(row_list)))
            stream.write("})")
        else:  # output json format to a file
            s = json.dumps((col_lists, rows))

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
        of = list(driver._objs)
        of.extend([c for c, meta in iteritems(driver._cons)
                   if not ('linear' in meta and meta['linear'])])
        wrt = list(driver._designvars)

        stream.write("\n\n")
        array_viz(J, problem, of, wrt, stream)

    return col_lists, rows


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
    column_lists, row_map = color_info

    desvars = problem.driver._designvars

    tot_colors = len(column_lists[0]) + len(column_lists) - 1
    tot_size = 0

    if problem._mode == 'fwd':
        for dv in desvars:
            tot_size += desvars[dv]['size']
    else:  # rev
        raise RuntimeError("rev mode currently not supported for simultaneous derivs.")

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
                                    show_jac=options.show_jac, stream=outfile)
        if sys.stdout.isatty():
            simul_coloring_summary(prob, color_info, stream=sys.stdout)

        exit()
    return _simul_coloring
