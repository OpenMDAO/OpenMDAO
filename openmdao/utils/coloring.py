"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

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


def _find_var_from_range(idx, ranges):
    # TODO: use bisection
    for start, end, name in ranges:
        if start <= idx <= end:
            return name, idx - start


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


def _find_disjoint(prob, mode='fwd', repeats=1, tol=1e-30):
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
        Tuple of the form (total_dv_offsets, total_res_offsets, final_jac)
    """
    # TODO: fix this to work in rev mode as well

    # clear out any old simul coloring info
    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    # remove any existing coloring metadata from dvs and responses
    for meta in chain(itervalues(prob.driver._designvars), itervalues(prob.driver._responses)):
        if 'simul_coloring' in meta:
            meta['simul_coloring'] = None
        if 'simul_map' in meta:
            meta['simul_map'] = None

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
    for n, meta in iteritems(prob.driver._cons):
        if not ('linear' in meta and meta['linear']):
            of.append(n)

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
    J = fullJ / np.linalg.norm(fullJ)

    boolJ = np.zeros(J.shape, dtype=bool)
    boolJ[J > tol] = True

    J = boolJ

    # from openmdao.utils.array_utils import array_viz
    # with open("arr_viz", 'w') as f:
    #     array_viz(J, prob=prob, of=of, wrt=wrt, stream=f)

    # find column and row ranges (inclusive) for dvs and responses respectively
    dv_offsets = []
    start = 0
    end = -1
    for name in wrt:
        end += prob.driver._designvars[name]['size']
        dv_offsets.append((start, end, name))
        start = end + 1

    res_offsets = []
    start = 0
    end = -1
    for name in of:
        end += responses[name]['size']
        res_offsets.append((start, end, name))
        start = end + 1

    total_dv_offsets = OrderedDict()
    total_res_offsets = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[], []])))

    # loop over each desvar and find disjoint column sets for all columns of that desvar
    for start, end, dv in dv_offsets:
        # skip desvars of size 1 since simul derivs will give no improvement
        if (end - start) == 0:
            continue

        disjoints = defaultdict(set)
        rows = {}
        for c1, c2 in combinations(range(start, end + 1), 2):  # loop over column pairs
            # 'and' two columns together. If we get all False, then columns have disjoint row sets
            if not np.any(J[:, c1] & J[:, c2]):
                disjoints[c1].add(c2)
                disjoints[c2].add(c1)
                if c1 not in rows:
                    rows[c1] = np.nonzero(J[:, c1])[0]
                if c2 not in rows:
                    rows[c2] = np.nonzero(J[:, c2])[0]

        full_disjoint = OrderedDict()
        seen = set()
        allrows = {}

        # sort largest to smallest disjoint column sets
        discols = sorted(disjoints.items(), key=lambda x: len(x[1]), reverse=True)

        for col, colset in discols:
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

        total_dv_offsets[dv] = tot_dv = OrderedDict()

        for color, cols in enumerate(full_disjoint.values()):
            tot_dv[color] = tot_dv_colors = []
            for c in sorted(cols):
                dvoffset = c - start
                tot_dv_colors.append(dvoffset)
                for crow in rows[c]:
                    res, resoffset = _find_var_from_range(crow, res_offsets)
                    dct = total_res_offsets[res][dv][color]
                    dct[0].append(resoffset)
                    dct[1].append(dvoffset)

    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    return total_dv_offsets, total_res_offsets, J


def get_simul_meta(problem, mode='fwd', repeats=1, tol=1.e-30, show_jac=False, stream=sys.stdout):
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

    dv_idxs, res_idxs, J = _find_disjoint(problem, mode=mode, tol=tol, repeats=repeats)
    all_colors = set()

    simul_colorings = {}
    simul_maps = {}

    for dv in dv_idxs:
        # negative colors will be iterated over individually, so start by filling the coloring array
        # with -1.  We then replace specific entries with positive colors which will be iterated
        # over as a group.
        coloring = np.full(driver._designvars[dv]['size'], -1)

        for color in dv_idxs[dv]:
            coloring[np.array(dv_idxs[dv][color], dtype=int)] = color
            all_colors.add(color)

        if np.any(coloring != -1):
            simul_colorings[dv] = list(coloring)

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

    if show_jac:
        of = list(driver._objs)
        of.extend([c for c, meta in iteritems(driver._cons)
                   if not ('linear' in meta and meta['linear'])])
        wrt = list(driver._designvars)

        stream.write("\n\n")
        array_viz(J, problem, of, wrt, stream)

    return simul_colorings, simul_maps


def simul_coloring_summary(problem, color_info, stream=sys.stdout):
    """
    Print a summary of simultaneous coloring info for the given problem and coloring metadata.

    Parameters
    ----------
    problem : Problem
        The Problem being analyzed.
    color_info : tuple of (simul_colorings, simul_maps)
        Coloring metadata.
    stream : file-like
        Where the output will go.
    """
    simul_colorings, simul_maps = color_info

    desvars = problem.driver._designvars
    constraints = problem.driver._cons
    responses = problem.driver._responses

    tot_colors = 0
    tot_size = 0

    dvwid = np.max([len(dv) for dv in desvars])
    dvtitle = "Variable"

    if len(dvtitle) > dvwid:
        dvwid = len(dvtitle)

    template = "{:<{w0}s}  {:>6d}  {:>6d}\n"

    stream.write("\n\n{:^{w0}s}  {:>6}  {:>6}\n".format(dvtitle, "Size", "Colors", w0=dvwid))
    stream.write("{:^{w0}s}  {:>6}  {:>6}\n".format('-' * dvwid, "----", "------", w0=dvwid))

    if problem._mode == 'fwd':
        for dv in desvars:
            if dv in simul_colorings:
                colors = set(simul_colorings[dv])
                if -1 in colors:
                    negs = len(np.nonzero(np.array(simul_colorings[dv]) < 0)[0])
                    ncolors = (negs + len(colors) - 1)
                else:
                    ncolors = len(colors)
            else:
                ncolors = desvars[dv]['size']

            size = desvars[dv]['size']
            stream.write(template.format(dv, size, ncolors, w0=dvwid))
            tot_colors += ncolors
            tot_size += desvars[dv]['size']
    else:  # rev
        raise RuntimeError("rev mode currently not supported for simultaneous derivs.")

    if not simul_colorings:
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
