"""
Various graph related utilities.
"""
from openmdao.utils.general_utils import ContainsAll
from itertools import combinations
import networkx as nx


def get_sccs_topo(graph):
    """
    Return strongly connected subsystems of the given Group in topological order.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph of Systems.

    Returns
    -------
    list of sets of str
        A list of strongly connected components in topological order.
    """
    # Tarjan's algorithm returns SCCs in reverse topological order, so
    # the list returned here is reversed.
    sccs = list(nx.strongly_connected_components(graph))
    sccs.reverse()
    return sccs


def all_connected_edges(graph, start):
    """

    Yield all downstream edges starting at the given node.

    Parameters
    ----------
    graph : network.DiGraph
        Graph being traversed.
    start : hashable object
        Identifier of the starting node.

    Yields
    ------
    list
        A list of all edges found when traversal starts at start.
    """
    stack = [start]
    visited = set(stack)
    while stack:
        src = stack.pop()
        for tgt in graph[src]:
            yield src, tgt
            if tgt not in visited:
                visited.add(tgt)
                stack.append(tgt)


def find_disjoint_vois(problem):
    """
    Report VOIs, grouped together based on which have shared dependent systems.

    VOIs in separate groups are disjoint and can have their derivatives
    computed simultaneously.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.

    Returns
    -------
    list of lists of str
        Contains dependent groups of variables of interest.
    """
    from openmdao.core.group import Group

    if problem._mode == 'fwd':
        vois = problem.model.get_design_vars().keys()
    else:
        vois = problem.model.get_responses().keys()
    relevant = problem.model._relevant

    # we need to remove all group names from the set of relevant systems for each
    # VOI.
    allgroups = set(s.pathname for s in
                    problem.model.system_iter(recurse=True, include_self=True,
                                              typ=Group))

    # Each node in this graph is a voi, and for each pair of nodes, if their dependency
    # sets are NOT disjoint, we connect them.  Connected components in the graph indicate
    # those vois that cannot have their derivatives solved for simultaneously.
    graph = nx.Graph()
    graph.add_nodes_from(vois)

    for a, b in combinations(vois, 2):
        rel_a = relevant[a]['@all'][1]
        if isinstance(rel_a, ContainsAll):
            continue
        else:
            rel_a = rel_a - allgroups

        rel_b = relevant[b]['@all'][1]
        if isinstance(rel_b, ContainsAll):
            continue
        else:
            rel_b = rel_b - allgroups

        if not rel_a.isdisjoint(rel_b):
            graph.add_edge(a, b)

    comps = [list(c) for c in nx.connected_components(graph)]

    # this returns groups of vois that come from different connected components,
    # because they are disjoint from each other (and would allow us to solve for their
    # total derivatives simultaneously).

    return comps
