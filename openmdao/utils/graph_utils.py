"""
Various graph related utilities.
"""
import networkx as nx
from openmdao.utils.general_utils import all_ancestors, common_subpath
from openmdao.utils.units import _find_unit


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


def get_cycle_tree(group):
    """
    Compute the tree of cycles for the given group.

    Parameters
    ----------
    group : <Group>
        The specified Group.

    Returns
    -------
    networkx.DiGraph
        The component graph for the given Group.
    dict
        A mapping of group path name to a tuple of the form
        (children, recursive_scc, unique_scc, scc_index, path, parent path or None).
    """
    from openmdao.core.group import iter_solver_info, Group

    G = group.compute_sys_graph(comps_only=True, add_edge_info=False)

    topo = get_sccs_topo(G)
    topsccs = [s for s in topo if len(s) > 1]
    common_paths = [common_subpath(s) for s in topsccs]
    cpathdict = {}
    for cpath, s in zip(common_paths, topsccs):
        if cpath not in cpathdict:
            cpathdict[cpath] = []
        cpathdict[cpath].append(s)

    topname = group.pathname
    group_tree_dict = {}
    for cpath, cpsccs in cpathdict.items():
        group_tree_dict[cpath] = [([], scc, set(scc), i, cpath, None)
                                  for i, scc in enumerate(cpsccs)]

    it = group._sys_tree_visitor(iter_solver_info, predicate=lambda s: isinstance(s, Group),
                                 yield_none=False)
    for tup in sorted(it, key=lambda x: (x[0].count('.'), len(x[0]))):
        path = tup[0]
        if not tup[2]:  # no sccs
            continue
        for ans in all_ancestors(path):
            if ans in group_tree_dict:
                parent_tree = group_tree_dict[ans]
                break
        else:
            parent_tree = group_tree_dict[topname]

        tree = group_tree_dict[path] if path in group_tree_dict else None

        prefix = path + '.' if path else ''
        for children, parent_scc, unique, _, parpath, _ in parent_tree:
            if prefix:
                matching_comps = [c for c in parent_scc if c.startswith(prefix)]
            else:
                matching_comps = parent_scc

            if matching_comps:
                subgraph = G.subgraph(matching_comps)
                sub_sccs = [s for s in get_sccs_topo(subgraph) if len(s) > 1]
                for sub_scc in sub_sccs:
                    if not sub_scc.isdisjoint(parent_scc) and sub_scc != parent_scc:
                        if tree is None:
                            group_tree_dict[path] = tree = ([])
                        tree.append(([], sub_scc, set(sub_scc), len(tree), path, parpath))
                        children.append(tree[-1])
                        # remmove the childs scc comps from the parent 'unique' scc
                        unique.difference_update(sub_scc)

    return G, group_tree_dict


def print_cycle_tree(group):
    """
    Print the tree of cycles for the given group.

    Parameters
    ----------
    group : <Group>
        The specified Group.
    """
    G, group_tree_dict = get_cycle_tree(group)

    def _print_tree(node, nscc, indent=''):
        children, scc, unique, i, path, _ = node
        print(indent, f"cycle {i + 1} of {nscc} for '{path}'")
        for u in unique:
            print(indent, f"  {u}")
        if children:
            for tup in children:
                _print_tree(tup, len(group_tree_dict[tup[4]]), indent + '  ')

    for lst in group_tree_dict.values():
        for _, _, _, idx, _, parpath in lst:
            if parpath is None:  # this is a top level scc
                _print_tree(lst[idx], len(lst))


def get_unresolved_knowns(graph, meta_name, nodes):
    """
    Return all unresolved nodes with known shape.

    Unresolved means that the node has known shape and at least one neighbor
    with unknown shape.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph containing all variables with shape info.
    meta_name : str
        The name of the node metadata variable to check.
    nodes : set of str
        Set of nodes to check.  All nodes must be in the graph.

    Returns
    -------
    set of str
        Set of nodes with known shape but at least one neighbor with unknown shape.
    """
    gnodes = graph.nodes
    unresolved = set()
    for node in nodes:
        if gnodes[node][meta_name] is not None:  # node has known shape
            for succ in graph.successors(node):
                if gnodes[succ][meta_name] is None:
                    unresolved.add(node)
                    break
            for pred in graph.predecessors(node):
                if gnodes[pred][meta_name] is None:
                    unresolved.add(node)
                    break

    return unresolved


def is_unresolved(graph, node, meta_name):
    """
    Return True if the given node is unresolved.

    Unresolved means that the node has at least one neighbor with unknown shape.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph containing all variables with shape info.
    node : str
        Node to check.
    meta_name : str
        The name of the node metadata variable to check.

    Returns
    -------
    bool
        True if the node is unresolved.
    """
    nodes = graph.nodes
    for s in graph.successors(node):
        if nodes[s][meta_name] is None:
            return True
    for p in graph.predecessors(node):
        if nodes[p][meta_name] is None:
            return True
    return False


def are_connected(graph, start, end):
    """
    Return True if the given source and target are connected.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph of Systems and variables.
    start : str
        Name of the starting node.
    end : str
        Name of the ending node.

    Returns
    -------
    bool
        True if the given source and target are connected.
    """
    if start in graph and end in graph:
        successors = graph.successors

        stack = [start]
        visited = set(stack)

        while stack:
            start = stack.pop()
            for node in successors(start):
                if node == end:
                    return True
                if node not in visited:
                    visited.add(node)
                    stack.append(node)

    return False


def get_active_edges(graph, knowns, meta_name):
    """
    Return all active single edges and active multi nodes.

    Active edges are those that are connected on one end to a variable with 'known' metadata
    and on the other end to a variable with 'unknown' metadata.  Active nodes are those that
    have 'unknown' metadata but are connected to a variable with 'known' metadata.

    Single edges correspond to 'x_by_conn' and 'copy_x' connections.
    Multi nodes are variables that have 'compute_x' set to True so they
    connect to multiple nodes of the opposite io type in a component. For example
    a 'compute_shape' output variable will connect to all inputs in the component and
    each of those edges will be labeled as 'multi'. So a multi node is a node that
    has 'multi' incoming edges.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph containing all variables with known/unknown info.
    knowns : list of str
        List of nodes with 'known' metadata.
    meta_name : str
        The name of the node metadata to check for 'known'/'unknown' status.

    Returns
    -------
    active_single_edges : set of (str, str)
        Set of active 'single' edges (for copy_x and x_by_conn).
    computed_nodes : set of str
        Set of active nodes with 'multi' edges (for compute_x).
    """
    nodes = graph.nodes
    edges = graph.edges
    active_single_edges = set()
    computed_nodes = set()

    for known in knowns:
        for succ in graph.successors(known):
            if nodes[succ][meta_name] is None:
                if edges[known, succ]['multi']:
                    computed_nodes.add(succ)
                else:
                    active_single_edges.add((known, succ))

        # We don't need to loop over predecessors here because this graph is not a dataflow
        # graph where data always flows from outputs to inputs.  In this case, the direction of
        # each edge is determined by the presence of shape_by_conn, etc., so for example, if an
        # output has shape_by_conn, then this graph has an edge from the connected input to
        # that output, which is the opposite direction of the edge between the same two variables
        # in our typical dataflow graph.  Since in this function we always start at 'known'
        # nodes, we only need to check their successors.

    return active_single_edges, computed_nodes


def meta2node_data(meta, to_extract):
    """
    Return a dict containing select metadata for the given variable.

    Parameters
    ----------
    meta : dict
        Metadata for the variable.
    to_extract : tuple of str
        Tuple of metadata names to extract.

    Returns
    -------
    dict
        Dict containing select metadata for the variable.
    """
    return {k: meta[k] for k in to_extract}


_shape_extract = ('distributed', 'shape', 'compute_shape', 'shape_by_conn', 'copy_shape')
_units_extract = ('units', 'compute_units', 'units_by_conn', 'copy_units')


def add_shape_node(graph, name, io, meta):
    """
    Add a shape node to the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to add the shape node to.
    name : str
        Name of the shape node.
    io : str
        Input or output.
    meta : dict
        Metadata for the variable.
    """
    kwargs = {k: meta[k] for k in _shape_extract}
    if io == 'input':
        kwargs['src_indices'] = meta.get('src_indices')
        kwargs['flat_src_indices'] = meta.get('flat_src_indices')
    graph.add_node(name, io=io, **kwargs)


def add_units_node(graph, name, io, meta):
    """
    Add a units node to the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to add the units node to.
    name : str
        Name of the units node.
    io : str
        Input or output.
    meta : dict
        Metadata for the variable.
    """
    kwargs = {k: meta[k] for k in _units_extract}
    if kwargs['units'] is not None:
        # Turn units data into a PhysicalUnit so units of expressions can be computed.
        kwargs['units'] = _find_unit(kwargs['units'])
    graph.add_node(name, io=io, **kwargs)
