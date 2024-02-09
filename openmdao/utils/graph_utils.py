"""
Various graph related utilities.
"""
import networkx as nx
from fnmatch import fnmatchcase


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


def get_out_of_order_nodes(graph, orders):
    """
    Return a list of nodes that are out of order.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph of Systems.
    orders : dict
        A dict of order values keyed by node name.

    Returns
    -------
    list of sets of str
        A list of strongly connected components in topological order.
    list of str
        A list of nodes that are out of order.
    """
    strongcomps = get_sccs_topo(graph)

    out_of_order = []
    for strongcomp in strongcomps:
        for u, v in graph.edges(strongcomp):
            # for any connection between a system in this strongcomp and a system
            # outside of it, the target must be ordered after the source.
            if u in strongcomp and v not in strongcomp and orders[u] > orders[v]:
                out_of_order.append((u, v))

    return strongcomps, out_of_order


def graph_grep(graph, patterns, search_nodes=True, search_edges=False):
    """
    Search the graph for nodes or edges that match any of the given patterns and display them.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph of Systems.
    patterns : list of str
        A list of patterns to search for.
    search_nodes : bool
        If True, search node names.
    search_edges : bool
        If True, search edge names.
    """
    nodeiter = graph.nodes()
    edgeiter = graph.edges()

    node_matches = set()
    edge_matches = set()

    for pattern in patterns:
        if search_nodes:
            node_matches.update(n for n in nodeiter if fnmatchcase(n, pattern))
        if search_edges:
            edge_matches.update(e for e in edgeiter if fnmatchcase(e[0], pattern) or
                                fnmatchcase(e[1], pattern))

    if node_matches:
        print("Matching nodes:")
        for node in sorted(node_matches):
            print("  ", node)
        print()

    if edge_matches:
        print("Matching edges:")
        for edge in sorted(edge_matches):
            print("  ", edge)
        print()
