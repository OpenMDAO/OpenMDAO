"""A module containing various configuration checks for an OpenMDAO Problem."""

import logging

import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from six import iteritems
import networkx as nx

from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.devtools.compat import abs_varname_iter, system_iter


def check_config(problem, logger=None):
    """Perform optional error checks on a Problem.

    Args
    ----
    problem : Problem
        The Problem being checked.

    logger : object
        Logging object.

    """
    if logger is None:
        logger = logging.getLogger("config_check")
        console = logging.StreamHandler()
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    root = problem.root

    _check_hanging_inputs(problem, logger)
    _check_cycles(root, logger)


def compute_sys_graph(group, input_src_ids, comps_only=False):
    """Compute a dependency graph for subsystems in the given group.

    Args
    ----
    group : <Group>
        The Group we're computing the graph for.

    input_src_ids : ndarray of int
        Array containing global variable ids for sources of the inputs
        indicated by the index into the array.

    comps_only : bool (False)
        If True, return a graph of all Components within the given group
        or any of its descendants. No sub-groups will be included. Otherwise,
        a graph containing only direct children (both Components and Groups)
        of the group will be returned.

    Returns
    -------
    DiGraph
        A directed graph containing names of subsystems and their connections.

    """
    if comps_only:
        subsystems = list(system_iter(group, recurse=True, typ=Component))
    else:
        subsystems = group._subsystems_allprocs

    nsubs = len(subsystems)

    i_start, i_end = group._variable_allprocs_range['input']
    o_start, o_end = group._variable_allprocs_range['output']

    # mapping arrays to find the system ID given the variable ID
    invar2sys = numpy.empty(i_end - i_start, dtype=int)
    outvar2sys = numpy.empty(o_end - o_start, dtype=int)

    for i, s in enumerate(subsystems):
        start, end = s._variable_allprocs_range['input']
        invar2sys[start - i_start:end - i_start] = i

        start, end = s._variable_allprocs_range['output']
        outvar2sys[start - o_start:end - o_start] = i

    graph = nx.DiGraph()

    for in_id, src_id in enumerate(input_src_ids):
        if (src_id != -1 and (o_start <= src_id < o_end) and
                (i_start <= in_id < i_end)):
            # offset the ids to index into our var2sys arrays
            graph.add_edge(subsystems[outvar2sys[src_id - o_start]].path_name,
                           subsystems[invar2sys[in_id - i_start]].path_name)

    return graph


def get_sccs(group, comps_only=False):
    """Return strongly connected subsystems of the given Group.

    Args
    ----
    group : <Group>
        The strongly connected components will be computed for this Group.

    comps_only : bool (False)
        If True, the graph used to compute strongly connected components
        will contain all Components within the given group or any of its
        descendants and no sub-groups will be included. Otherwise, the graph
        used will contain only direct children (both Components and Groups)
        of the given group.

    Returns
    -------
    list of sets of str
        A list of strongly connected components in topological order.
    """
    graph = compute_sys_graph(group, group._sys_assembler._input_src_ids,
                              comps_only=comps_only)

    # Tarjan's algorithm returns SCCs in reverse topological order, so
    # the list returned here is reversed.
    sccs = list(nx.strongly_connected_components(graph))
    sccs.reverse()
    return sccs


def _check_cycles(group, logger):
    """Report any cycles found in any Group to the logger.

    Args
    ----
    group : <Group>
        The Group being checked for cycles.

    logger : object
        The object that manages logging output.

    Returns
    -------
    list of sets
        A list of strongly connected components of the system
        dependency grapy of the given group. SCCs are sorted in
        topological order.

    """
    for system in system_iter(group, include_self=True, recurse=True,
                              typ=Group):
        sccs = get_sccs(system)
        cycles = [sorted(s) for s in sccs if len(s) > 1]
        if cycles:
            logger.warning("Group '%s' has the following cycles: %s" %
                           (system.path_name, cycles))

    # in case the sccs are needed elsewhere, just return it so it doesn't
    # need to be recomuted.
    return sccs


def _check_hanging_inputs(problem, logger):
    """Issue a logger warning if any inputs are not connected.

    Args
    ----
    problem : <Problem>
        The problem being checked.

    logger : object
        The object that managers logging output.

    """
    input_src_ids = problem._assembler._input_src_ids

    hanging = sorted([
        name for i, name in enumerate(abs_varname_iter(problem.root, 'input',
                                                       local=False)) if
                                                       input_src_ids[i] == -1
    ])

    if hanging:
        logger.warning("The following inputs are not connected: %s." % hanging)
