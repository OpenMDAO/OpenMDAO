"""A module containing various configuration checks for an OpenMDAO Problem."""

import logging

import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from openmdao.core.group import Group
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
        logger = logging.getLogger()
        console = logging.StreamHandler()
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    _check_hanging_inputs(problem, logger)
    _check_cycles(problem, logger)


def compute_group_sys_graph(group, input_src_ids):
    """Compute a dependency graph for subsystems in the given group.

    Args
    ----
    group : <Group>
        The Group we're computing the graph for.

    input_src_ids : ndarray of int
        Array containing global variable ids for sources of the inputs
        indicated by the index into the array.

    Returns
    -------
    csr_matrix
        A graph in the form of a sparse (CSR) adjacency matrix.

    """
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

    rows = []
    cols = []

    for in_id, src_id in enumerate(input_src_ids):
        if (src_id != -1 and (o_start <= src_id < o_end) and
                (i_start <= in_id < i_end)):
            # offset the ids to index into our var2sys arrays
            rows.append(outvar2sys[src_id - o_start])
            cols.append(invar2sys[in_id - i_start])

    data = numpy.ones(len(rows))

    return csr_matrix((data, (rows, cols)), shape=(nsubs, nsubs))


def _check_cycles(problem, logger):
    group_sccs = {}
    for system in system_iter(problem.root, include_self=True, recurse=True):
        if isinstance(system, Group):
            sccs = []
            subs = system._subsystems_allprocs
            graph = compute_group_sys_graph(system,
                                            problem._assembler._input_src_ids)
            num_sccs, labels = connected_components(graph, connection='strong')

            for i in range(num_sccs):
                # find systems in SCC i
                connected_systems = numpy.where(labels == i)[0]
                if connected_systems.size > 1:
                    sccs.append([
                        subs[i].path_name for i in connected_systems
                    ])

            if sccs:
                group_sccs[system.path_name] = sccs

    if group_sccs:
        for gname, gsccs in iteritems(group_sccs):
            logger.warning("Group '%s' has the following cycles: %s" % (gname,
                                                                        gsccs))


def _check_hanging_inputs(problem, logger):
    """Warn if any inputs are not connected."""
    input_src_ids = problem._assembler._input_src_ids

    hanging = sorted([
        name for i, name in enumerate(abs_varname_iter(problem.root, 'input',
                                                       local=False)) if
                                                       input_src_ids[i] == -1
    ])

    if hanging:
        logger.warning("The following inputs are not connected: %s." % hanging)
