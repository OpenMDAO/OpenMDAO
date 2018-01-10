"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import sys
import json

from collections import OrderedDict, defaultdict
from itertools import combinations
from numbers import Integral

from six import iteritems
from six.moves import range

import numpy as np
from numpy.random import rand

from openmdao.jacobians.jacobian import Jacobian
from openmdao.jacobians.assembled_jacobian import AssembledJacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.matrix import sparse_types


def _find_var_from_range(idx, ranges):
    # TODO: use bisection
    for start, end, name in ranges:
        if start <= idx <= end:
            return name, idx - start


def _wrapper_set_abs(jac, set_abs, key, subjac, tol):
    info, shape = jac._subjacs_info[key]
    if info['rows'] is not None:  # list form
        spread = np.max(subjac) - np.min(subjac)
        if spread < .01:
            spread = 1.0
        subjac = data = rand(info['rows'].size) * spread - spread / 2.0
    elif isinstance(info['value'], sparse_types):  # sparse
        spread = np.max(subjac.data) - np.min(subjac.data)
        if spread < .01:
            spread = 1.0
        subjac = subjac.copy()
        subjac.data = data = rand(subjac.data.size) * spread - spread / 2.0
    else:   # dense
        spread = np.max(subjac) - np.min(subjac)
        if spread < .01:
            spread = 1.0
        subjac = data = rand(*(subjac.shape)) * spread - spread / 2.0

    data[data < tol] += spread
    data[data > -tol] -= spread

    return set_abs(key, subjac)


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
        Tuple of dicts total_dv_offsets and total_res_offsets.
    """
    # TODO: fix this to work in rev mode as well

    jac = prob.model._jacobian
    set_abs = jac._set_abs

    # replace existing jacobian set_abs with ours that replaces all subjacs with random numbers
    jac._set_abs = lambda key, subjac: _wrapper_set_abs(jac, set_abs, key, subjac, tol)

    # clear out any old simul coloring info
    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    prob.setup(mode=mode)
    prob.run_model()

    desvars = prob.driver._designvars
    responses = prob.driver._responses
    of = list(prob.driver._objs)
    wrt = list(desvars)

    if not of or not wrt:
        raise RuntimeError("Sparsity structure cannot be computed without declaration of design "
                           "variables and responses.")

    # remove linear constraints from consideration
    for n, meta in iteritems(prob.driver._cons):
        if not ('linear' in meta and meta['linear']):
            of.append(n)

    sumJ = None
    for i in range(repeats):
        J = prob.driver._compute_totals(return_format='array', of=of, wrt=wrt)
        if sumJ is None:
            sumJ = np.zeros(J.shape, dtype=bool)
            sumJ[J < -tol] = True
            sumJ[J > tol] = True
        else:
            sumJ |= np.logical_and(J < -tol, J > tol)

    J = sumJ

    # from openmdao.utils.array_utils import array_viz
    # with open("arr_viz", 'w') as f:
    #     #array_viz(J, stream=f)
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

    total_dv_offsets = defaultdict(lambda: defaultdict(list))
    total_res_offsets = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[], []])))

    # loop over each desvar and find disjoint column sets for all columns of that desvar
    for start, end, dv in dv_offsets:
        if (end + 1 - start) < 2:
            continue

        disjoints = defaultdict(set)
        rows = {}
        for c1, c2 in combinations(range(start, end + 1), 2):  # loop over column pairs
            result = J[:, c1] | J[:, c2]
            if np.max(result) <= 1.0:
                disjoints[c1].add(c2)
                disjoints[c2].add(c1)
                if c1 not in rows:
                    rows[c1] = set(np.nonzero(J[:, c1])[0])
                if c2 not in rows:
                    rows[c2] = set(np.nonzero(J[:, c2])[0])

        full_disjoint = {}
        seen = set()
        allrows = {}
        discols = sorted(disjoints.items(), key=lambda x: len(x[1]), reverse=True)

        for col, s in discols:
            if col in seen or col in full_disjoint:
                continue
            seen.add(col)
            allrows[col] = set(rows[col])
            full_disjoint[col] = set([col])
            for other_col in s:
                if other_col not in seen and not allrows[col].intersection(rows[other_col]):
                    seen.add(other_col)
                    full_disjoint[col].add(other_col)
                    allrows[col].update(rows[other_col])

        for color, cols in enumerate(full_disjoint.values()):
            for c in sorted(cols):
                dvoffset = c - start
                total_dv_offsets[dv][color].append(dvoffset)
                for crow in rows[c]:
                    res, resoffset = _find_var_from_range(crow, res_offsets)
                    dct = total_res_offsets[res][dv][color]
                    dct[0].append(resoffset)
                    dct[1].append(dvoffset)

    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    return total_dv_offsets, total_res_offsets


def get_simul_meta(problem, mode='fwd', repeats=1, tol=1.e-30, stream=sys.stdout):
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

    dv_idxs, res_idxs = _find_disjoint(problem, mode=mode, tol=tol, repeats=repeats)
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

        simul_colorings[dv] = list(coloring)

    for res in res_idxs:
        simul_map = {}
        for dv in res_idxs[res]:
            simul_map[dv] = {c: v for c, v in iteritems(res_idxs[res][dv])
                             if c in all_colors}
            if not simul_map[dv]:
                del simul_map[dv]

        if simul_map:
            simul_maps[res] = simul_map

    if stream is not None:
        s = json.dumps((simul_colorings, simul_maps))

        # do a little pretty printing since the built-in json pretty printing stretches
        # the output vertically WAY too much.
        s = s.replace(',"', ',\n    "')
        s = s.replace(', "', ',\n    "')
        s = s.replace('{"', '{\n    "')

        stream.write(s)
        stream.write("\n")

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
    responses = problem.driver._responses

    stream.write("\n\nColoring Summary\n")

    tot_colors = 0
    tot_size = 0
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

            stream.write("%s num colors: %d   size: %d\n" % (dv, ncolors, desvars[dv]['size']))
            tot_colors += ncolors
            tot_size += desvars[dv]['size']
    else:  # rev
        raise RuntimeError("rev mode currently not supported for simultaneous derivs.")

    if not simul_colorings:
        stream.write("No simultaneous derivative solves are possible in this configuration.\n")
    else:
        stream.write("Total colors vs. total size: %d vs %d\n" % (tot_colors, tot_size))


def _simul_coloring_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao simul_coloring' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', action='store', dest='outfile', help='output file.')
    parser.add_argument('-n', action='store', dest='num_jacs', default=1, type=int,
                        help='number of times to repeat total deriv computation.')
    parser.add_argument('-t', action='store', dest='tolerance', default=1.e-30, type=float,
                        help='tolerance used to determine if an array entry is nonzero.')


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
                                    stream=outfile)
        if sys.stdout.isatty():
            simul_coloring_summary(prob, color_info, stream=sys.stdout)

        exit()
    return _simul_coloring
