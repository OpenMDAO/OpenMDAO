"""A module containing various configuration checks for an OpenMDAO Problem."""

import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import networkx as nx
from six import iteritems

from openmdao.core.group import Group
from openmdao.core.component import Component

# when setup is called multiple times, we need this to prevent adding
# another handler to the config_check logger each time (if logger arg to check_config is None)
_set_logger = None


def check_config(problem, logger=None):
    """
    Perform optional error checks on a Problem.

    Parameters
    ----------
    problem : Problem
        The Problem being checked.

    logger : object
        Logging object.
    """
    global _set_logger
    if logger is None:
        if _set_logger is None:
            logger = logging.getLogger("config_check")
            _set_logger = logger
            console = logging.StreamHandler()
            # set a format which is simpler for console use
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            # tell the handler to use this format
            console.setFormatter(formatter)
            console.setLevel(logging.INFO)
            logger.addHandler(console)
        else:
            logger = _set_logger

    _check_hanging_inputs(problem, logger)
    _check_dataflow(problem.model, logger)


def compute_sys_graph(group, input_srcs, comps_only=False):
    """
    Compute a dependency graph for subsystems in the given group.

    Parameters
    ----------
    group : <Group>
        The Group we're computing the graph for.

    input_srcs : {}
        dict containing global variable abs names for sources of the inputs.

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
        subsystems = list(group.system_iter(recurse=True, typ=Component))
    else:
        subsystems = group._subsystems_allprocs

    i_start = group._ext_num_vars['input'][0]
    i_end = group._ext_num_vars['input'][0] + group._num_var['input']
    o_start = group._ext_num_vars['output'][0]
    o_end = group._ext_num_vars['output'][0] + group._num_var['output']

    # mapping arrays to find the system ID given the variable ID
    invar2sys = np.empty(group._num_var['input'], dtype=int)
    outvar2sys = np.empty(group._num_var['output'], dtype=int)

    for i, s in enumerate(subsystems):
        start = s._ext_num_vars['input'][0]
        end = s._ext_num_vars['input'][0] + s._num_var['input']
        invar2sys[start - i_start:end - i_start] = i

        start = s._ext_num_vars['output'][0]
        end = s._ext_num_vars['output'][0] + s._num_var['output']
        outvar2sys[start - o_start:end - o_start] = i

    graph = nx.DiGraph()

    indices = group._var_allprocs_abs2idx
    for in_abs, src_abs in iteritems(input_srcs):
        if src_abs is not None:
            src_id = indices['output'][src_abs] + group._ext_num_vars['output'][0]
            in_id = indices['input'][in_abs] + group._ext_num_vars['input'][0]
            if ((o_start <= src_id < o_end) and (i_start <= in_id < i_end)):
                graph.add_edge(subsystems[outvar2sys[src_id - o_start]].pathname,
                               subsystems[invar2sys[in_id - i_start]].pathname)

    return graph


def get_sccs(group, comps_only=False):
    """
    Return strongly connected subsystems of the given Group.

    Parameters
    ----------
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
    graph = compute_sys_graph(group, group._conn_global_abs_in2out,
                              comps_only=comps_only)

    # Tarjan's algorithm returns SCCs in reverse topological order, so
    # the list returned here is reversed.
    sccs = list(nx.strongly_connected_components(graph))
    sccs.reverse()
    return sccs


def _check_dataflow(group, logger):
    """
    Report any cycles and out of order Systems to the logger.

    Parameters
    ----------
    group : <Group>
        The Group being checked for dataflow issues.

    logger : object
        The object that manages logging output.
    """
    for system in group.system_iter(include_self=True, recurse=True, typ=Group):
        sccs = get_sccs(system)
        cycles = [sorted(s) for s in sccs if len(s) > 1]
        cycle_idxs = {}

        if cycles:
            logger.warning("Group '%s' has the following cycles: %s" %
                           (system.pathname, cycles))
            for i, cycle in enumerate(cycles):
                # keep track of cycles so we can detect when a system in
                # one cycle is out of order with a system in a different cycle.
                for s in cycle:
                    cycle_idxs[s] = i

        ubcs = _get_out_of_order_subs(system, system._conn_global_abs_in2out)

        for tgt_system, src_systems in sorted(ubcs.items()):
            keep_srcs = []

            for src_system in src_systems:
                if not (src_system in cycle_idxs and
                        tgt_system in cycle_idxs and
                        cycle_idxs[tgt_system] == cycle_idxs[src_system]):
                    keep_srcs.append(src_system)

            if keep_srcs:
                logger.warning("System '%s' executes out-of-order with "
                               "respect to its source systems %s" %
                               (tgt_system, sorted(keep_srcs)))


def _get_out_of_order_subs(group, input_srcs):
    """
    Return Systems that are executed out of dataflow order.

    Parameters
    ----------
    group : <Group>
        The Group where we're checking subsystem order.

    input_srcs : {}
        dict containing variable abs names for sources of the inputs.
        This describes all variable connections, either explicit or implicit,
        in the entire model.

    Returns
    -------
    dict
        A dict mapping names of target Systems to a list of names of their
        source Systems that execute after them.
    """
    subsystems = group._subsystems_allprocs

    i_start = group._ext_num_vars['input'][0]
    i_end = group._ext_num_vars['input'][0] + group._num_var['input']
    o_start = group._ext_num_vars['output'][0]
    o_end = group._ext_num_vars['output'][0] + group._num_var['output']

    # mapping arrays to find the system ID given the variable ID
    invar2sys = np.empty(i_end - i_start, dtype=int)
    outvar2sys = np.empty(o_end - o_start, dtype=int)

    for i, s in enumerate(subsystems):
        start = s._ext_num_vars['input'][0]
        end = s._ext_num_vars['input'][0] + s._num_var['input']
        invar2sys[start - i_start:end - i_start] = i

        start = s._ext_num_vars['output'][0]
        end = s._ext_num_vars['output'][0] + s._num_var['output']
        outvar2sys[start - o_start:end - o_start] = i

    indices = group._var_allprocs_abs2idx
    ubcs = {}
    for in_abs, src_abs in iteritems(input_srcs):
        if src_abs is not None:
            src_id = indices['output'][src_abs] + group._ext_num_vars['output'][0]
            in_id = indices['input'][in_abs] + group._ext_num_vars['input'][0]
            if ((o_start <= src_id < o_end) and (i_start <= in_id < i_end)):
                # offset the ids to index into our var2sys arrays
                src_sysID = outvar2sys[src_id - o_start]
                tgt_sysID = invar2sys[in_id - i_start]
                if (src_sysID > tgt_sysID):
                    src_sys = subsystems[src_sysID].pathname
                    tgt_sys = subsystems[tgt_sysID].pathname
                    ubcs.setdefault(tgt_sys, []).append(src_sys)

    return ubcs


def _check_hanging_inputs(problem, logger):
    """
    Issue a logger warning if any inputs are not connected.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.

    logger : object
        The object that managers logging output.
    """
    input_srcs = problem.model._conn_global_abs_in2out

    hanging = sorted([
        name
        for name in problem.model._var_allprocs_abs_names['input']
        if name not in input_srcs
    ])

    if hanging:
        logger.warning("The following inputs are not connected: %s." % hanging)
