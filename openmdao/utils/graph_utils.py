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


def get_hybrid_graph(connections):
    """
    Return a graph of all variables and components in the model.

    Each component is connected each of its input and output variables, and
    those variables are connected to other variables based on the connections
    in the model.

    Parameters
    ----------
    connections : dict
        Dictionary of connections in the model, of the form {tgt: src}.

    Returns
    -------
    networkx.DiGraph
        Graph of all variables and components in the model.
    """
    # Create a hybrid graph with components and all connected vars.  If a var is connected,
    # also connect it to its corresponding component.  This results in a smaller graph
    # (fewer edges) than would be the case for a pure variable graph where all inputs
    # to a particular component would have to be connected to all outputs from that component.
    graph = nx.DiGraph()
    for tgt, src in connections.items():
        if src not in graph:
            graph.add_node(src, type_='out')

        graph.add_node(tgt, type_='in')

        src_sys, _, _ = src.rpartition('.')
        graph.add_edge(src_sys, src)

        tgt_sys, _, _ = tgt.rpartition('.')
        graph.add_edge(tgt, tgt_sys)

        graph.add_edge(src, tgt)

    return graph
