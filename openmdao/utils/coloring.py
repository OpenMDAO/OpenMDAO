"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import sys

from collections import OrderedDict, defaultdict, namedtuple
from itertools import product
import logging

from six import iteritems, iterkeys, itervalues
from six.moves import range

import numpy as np

from openmdao.core.component import Component
from openmdao.jacobians.assembled_jacobian import DenseJacobian
from openmdao.matrices.dense_matrix import DenseMatrix

from openmdao.utils.name_maps import rel_key2abs_key, key2abs_key


class _SimulJacobian(DenseJacobian):
    """
    Assemble dense global <Jacobian> for use in disjoint row/column analysis.

    NOTE: DO NOT USE this jacobian for any actual calculations.  This is used purely to determine
    sparsity of the total jacobian.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(_SimulJacobian, self, **kwargs).__init__()
        self.options['matrix_class'] = DenseMatrix

    def __setitem__(self, key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        abs_key = key2abs_key(self._system, key)
        if abs_key is not None:

            # You can only set declared subjacobians.
            if abs_key not in self._subjacs_info:
                msg = 'Variable name pair ("{}", "{}") must first be declared.'
                raise KeyError(msg.format(key[0], key[1]))

            # get a version of the subjac corresponding to abs_key and fill with 1's
            info, shape = self._subjacs_info[abs_key]
            if info['rows'] is not None:  # list form
                subjac = np.ones(info['rows'].size, dtype=bool)
            elif isinstance(info['value'], sparse_types):  # sparse
                subjac = subjac.copy()
                subjac.data = np.ones(subjac.data.size, dtype=bool)
            else:   # dense
                print("DENSE:", abs_key, subjac.shape)
                subjac = np.ones(subjac.shape, dtype=bool)
            self._set_abs(abs_key, subjac)
        else:
            msg = 'Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(key[0], key[1]))


def _find_var_from_range(idx, ranges):
    # TODO: use bisection
    for start, end, name in ranges:
        if start <= idx <= end:
            return name, idx - start


def _find_disjoint(prob, mode='fwd', tol=1e-10):
    """
    Find all sets of disjoint columns in the total jac and their corresponding rows.

    Parameters
    ----------
    prob : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
    tol : float
        Tolerance on values in jacobian.  Anything smaller in magnitude will be set to 0.0.

    Returns
    -------
    tuple
        Tuple of dicts total_dv_offsets and total_res_offsets.
    """
    # TODO: fix this to work in rev mode as well

    from collections import defaultdict
    from itertools import combinations, product
    from openmdao.utils.array_utils import array_viz

    # prob.model.jacobian = _SimulJacobian()
    # clear out any old simul coloring info
    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    prob.setup(mode=mode)
    prob.run_model()

    of = list(prob.driver._objs)
    of_lin = []
    desvars = prob.driver._designvars
    responses = prob.driver._responses

    wrt = list(desvars)
    for n, meta in iteritems(prob.driver._cons):
        if 'linear' in meta and meta['linear']:
            of_lin.append(n)
        else:
            of.append(n)

    J = prob.driver._compute_totals(return_format='array', of=of, wrt=wrt)
    J[np.abs(J) < tol] = 0.0
    J[np.abs(J) >= tol] = 1.0

    # find column and row ranges (inclusive) for dvs and responses respectively
    dv_offsets = []
    start = 0
    end = -1
    for name in wrt:
        end += prob.driver._designvars[name]['size']
        dv_offsets.append((start, end, name))
        # print("dv range[%s] = %s" % (name, (start, end)))
        start = end + 1

    res_offsets = []
    start = 0
    end = -1
    for name in of:
        end += responses[name]['size']
        res_offsets.append((start, end, name))
        # print("res range[%s] = %s" % (name, (start, end)))
        start = end + 1

    total_dv_offsets = defaultdict(lambda: defaultdict(list))
    total_res_offsets = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], []))))

    # loop over each desvar and find disjoint column sets for all columns of that desvar
    for start, end, dv in dv_offsets:
        allcols = list(range(start, end + 1))
        if len(allcols) < 2:
            continue

        disjoints = defaultdict(set)
        rows = {}
        for c1, c2 in combinations(allcols, 2):  # loop over column pairs
            result = J[:, c1] + J[:, c2]
            if np.all(result <= 1.0):
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
            # print("\ncolor %d:" % color)
            for c in sorted(cols):
                dvoffset = c - start
                total_dv_offsets[dv][color].append(dvoffset)
                # print(dv, dvoffset, 'col', c)
                for crow in rows[c]:
                    res, resoffset = _find_var_from_range(crow, res_offsets)
                    # print("   ", res, resoffset, 'row', crow)
                    dct = total_res_offsets[res][dv][color]
                    dct[0].append(resoffset)
                    dct[1].append(dvoffset)

    prob.driver._simul_coloring_info = None
    prob.driver._res_jacs = {}

    return total_dv_offsets, total_res_offsets


def get_simul_meta(problem, mode='fwd', stream=sys.stdout):
    """
    Compute simultaneous derivative colorings for the given problem.

    Parameters
    ----------
    problem : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
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
    if not driver.supports['simultaneous_derivatives']:
        return {}, {}

    dv_idxs, res_idxs = _find_disjoint(problem, mode=mode)
    all_colors = set()

    simul_colorings = {}
    simul_maps = {}

    for dv in dv_idxs:
        # negative colors will be iterated over individually, so start by filling the coloring array
        # with -1.  We then replace specific entries with positive colors which will be iterated
        # over as a group.
        coloring = np.full(driver._designvars[dv]['size'], -1)
        # has_color = False
        for color in dv_idxs[dv]:
            # if len(dv_idxs[dv][color]) > 1:
            coloring[np.array(dv_idxs[dv][color], dtype=int)] = color
            all_colors.add(color)
            # has_color = True
        # if has_color:
        simul_colorings[dv] = list(coloring)

    for res in res_idxs:
        simul_map = {}
        for dv in res_idxs[res]:
            simul_map[dv] = {c: v for c, v in iteritems(res_idxs[res][dv]) if c in all_colors}
            if not simul_map[dv]:
                del simul_map[dv]

        if simul_map:
            simul_maps[res] = simul_map

    if stream is not None:
        stream.write("\n%s\n" % ((simul_colorings, simul_maps),))

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
            print(dv, 'num colors:', ncolors)
            tot_colors += ncolors
            tot_size += desvars[dv]['size']
    else:  # rev
        raise RuntimeError("rev mode currently not supported for simultaneous derivs.")

    stream.write("\nColoring Summary\n")
    # for dv, meta in iteritems(desvars):
    #     stream.write("DV: %s  %d\n" % (dv, meta['size']))
    # for res, meta in iteritems(responses):
    #     stream.write("Resp: %s  %d\n" % (res, meta['size']))

    stream.write("Total colors vs. total size: %d vs %d\n" % (tot_colors, tot_size))
