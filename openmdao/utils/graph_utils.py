"""
Various graph related utilities.
"""
import networkx as nx
from openmdao.utils.general_utils import all_ancestors, common_subpath


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
