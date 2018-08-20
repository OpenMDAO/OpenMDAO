"""
Various graph related utilities.
"""
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


def all_connected_nodes(graph, start):
    """
    Yield all downstream nodes starting at the given node.

    Parameters
    ----------
    graph : network.DiGraph
        Graph being traversed.
    start : hashable object
        Identifier of the starting node.

    Yields
    ------
    list
        A list of all nodes found when traversal starts at start.
    """
    stack = [start]
    visited = set(stack)
    yield start
    while stack:
        src = stack.pop()
        for tgt in graph[src]:
            yield tgt
            if tgt not in visited:
                visited.add(tgt)
                stack.append(tgt)
