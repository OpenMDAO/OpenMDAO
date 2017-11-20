"""A module containing various configuration checks for an OpenMDAO Problem."""

from collections import defaultdict
from six import iteritems

import networkx as nx
import numpy as np

from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.utils.graph_utils import get_sccs_topo
from openmdao.utils.logger_utils import get_logger


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
    logger = logger if logger else get_logger('check_config', use_format=True)

    _check_hanging_inputs(problem, logger)

    for system in problem.model.system_iter(include_self=True, recurse=True):
        # system specific check
        system.check_config(logger)
        # check dataflow within Group
        if isinstance(system, Group):
            _check_dataflow(system, logger)


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
    graph = group.compute_sys_graph(comps_only=False)
    sccs = get_sccs_topo(graph)
    sub2i = {sub.name: i for i, sub in enumerate(group._subsystems_allprocs)}
    cycles = [sorted(s, key=lambda n: sub2i[n]) for s in sccs if len(s) > 1]
    cycle_idxs = {}

    if cycles:
        logger.warning("Group '%s' has the following cycles: %s" %
                       (group.pathname, cycles))
        for i, cycle in enumerate(cycles):
            # keep track of cycles so we can detect when a system in
            # one cycle is out of order with a system in a different cycle.
            for s in cycle:
                if group.pathname:
                    s = '.'.join((group.pathname, s))
                cycle_idxs[s] = i

    ubcs = _get_out_of_order_subs(group, group._conn_global_abs_in2out)

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
        A dict mapping names of target Systems to a set of names of their
        source Systems that execute after them.
    """
    subsystems = group._subsystems_allprocs
    sub2i = {sub.name: i for i, sub in enumerate(subsystems)}
    glen = len(group.pathname.split('.')) if group.pathname else 0

    ubcs = defaultdict(set)
    for in_abs, src_abs in iteritems(input_srcs):
        if src_abs is not None:
            iparts = in_abs.split('.')
            oparts = src_abs.split('.')
            src_sys = oparts[glen]
            tgt_sys = iparts[glen]
            if (src_sys in sub2i and tgt_sys in sub2i and
                    (sub2i[src_sys] > sub2i[tgt_sys])):
                ubcs['.'.join(iparts[:glen + 1])].add('.'.join(oparts[:glen + 1]))

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
