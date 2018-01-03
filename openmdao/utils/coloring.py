"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import sys

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


def _wrapper_set_abs(jac, set_abs, key, subjac):
    info, shape = jac._subjacs_info[key]
    if info['rows'] is not None:  # list form
        subjac = rand(info['rows'].size)
    elif isinstance(info['value'], sparse_types):  # sparse
        subjac = subjac.copy()
        subjac.data = rand(subjac.data.size)
    else:   # dense
        subjac = rand(*(subjac.shape))

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
    jac._set_abs = lambda key, subjac: _wrapper_set_abs(jac, set_abs, key, subjac)

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
        absJ = np.abs(J)
        if sumJ is None:
            sumJ = absJ
        else:
            sumJ += absJ

    J = sumJ
    J[J < tol] = 0.0
    J[J >= tol] = 1.0

    # from openmdao.utils.array_utils import array_viz
    # array_viz(J)

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
    total_res_offsets = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], []))))

    # loop over each desvar and find disjoint column sets for all columns of that desvar
    for start, end, dv in dv_offsets:
        if (end + 1 - start) < 2:
            continue

        disjoints = defaultdict(set)
        rows = {}
        for c1, c2 in combinations(range(start, end + 1), 2):  # loop over column pairs
            result = J[:, c1] + J[:, c2]
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


def get_simul_meta(problem, mode='fwd', repeats=1, stream=sys.stdout):
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

    dv_idxs, res_idxs = _find_disjoint(problem, mode=mode, repeats=repeats)
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
            stream.write("%s num colors: %d\n" % (dv, ncolors))
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
        color_info = get_simul_meta(prob, repeats=options.num_jacs, stream=outfile)
        simul_coloring_summary(prob, color_info, stream=outfile)
        exit()
    return _simul_coloring
