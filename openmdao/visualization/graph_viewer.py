"""
Viewer graphs of a group's model hierarchy and connections.
"""
from itertools import chain

try:
    import pydot
except ImportError:
    pydot = None

import networkx as nx

from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.general_utils import all_ancestors
from openmdao.utils.file_utils import _load_and_exec
import openmdao.utils.hooks as hooks


# mapping of base system type to graph display properties
_base_display_map = {
    'ExplicitComponent': {
        'fillcolor': '"aquamarine3:aquamarine"',
        'style': 'filled',
        'shape': 'box',
    },
    'ImplicitComponent': {
        'fillcolor': '"lightblue:lightslateblue"',
        'style': 'filled',
        'shape': 'box',
    },
    'IndepVarComp': {
        'fillcolor': '"chartreuse2:chartreuse4"',
        'style': 'filled',
        'shape': 'box',
    },
    'Group': {
        'fillcolor': 'gray75',
        'style': 'filled',
        'shape': 'box',
    },
}


class GraphViewer(object):
    """
    A class for viewing the model hierarchy and connections in a group.

    Parameters
    ----------
    group : <Group>
        The Group with graph to be viewed.

    Attributes
    ----------
    _group : <Group>
        The Group with graph to be viewed.
    """

    def __init__(self, group):
        """
        Initialize the GraphViewer.

        Parameters
        ----------
        group : <Group>
            The Group with graph to be viewed.
        """
        self._group = group

    def write_graph(self, gtype='dataflow', recurse=True, show_vars=False,
                    display=True, show_boundary=False, exclude=(), outfile=None):
        """
        Use pydot to create a graphical representation of the specified graph.

        Parameters
        ----------
        gtype : str
            The type of graph to create. Options include 'system', 'component', 'nested',
            and 'dataflow'.
        recurse : bool
            If True, recurse into subsystems when gtype is 'dataflow'.
        show_vars : bool
            If True, show all variables in the graph. Only relevant when gtype is 'dataflow'.
        display : bool
            If True, pop up a window to view the graph.
        show_boundary : bool
            If True, include connections to variables outside the boundary of the Group.
        exclude : iter of str
            Iter of pathnames to exclude from the generated graph.
        outfile : str or None
            The name of the file to write the graph to.  The format is inferred from the extension.
            Default is None, which writes to '<system_path>_<gtype>_graph.html', where system_path
            is 'model' for the top level group, and any '.' in the pathname is replaced with '_'.

        Returns
        -------
        pydot.Dot or None
            The pydot graph that was created.
        """
        if pydot is None:
            raise RuntimeError(f"{self.msginfo}: write_graph requires pydot.  Install pydot using "
                               "'pip install pydot'. Note that pydot requires graphviz, which is a "
                               "non-Python application.\nIt can be installed at the system level "
                               "or via a package manager like conda.")

        group = self._group

        if gtype == 'tree':
            G = self._get_tree_graph(exclude)
        elif gtype == 'dataflow':
            if show_vars:
                # get dvs and responses so we can color them differently
                dvs = group.get_design_vars(recurse=True, get_sizes=False, use_prom_ivc=True)
                desvars = set(dvs)
                desvars.update(m['source'] for m in dvs.values())
                resps = group.get_responses(recurse=True, get_sizes=False, use_prom_ivc=True)
                responses = set(resps)
                responses.update(m['source'] for m in resps.values())

                prefix = group.pathname + '.' if group.pathname else ''
                lenpre = len(prefix)

                G = group._problem_meta['model_ref']()._get_dataflow_graph()

                if not recurse:
                    # layout graph from left to right
                    gname = 'model' if group.pathname == '' else group.pathname
                    G.graph['graph'] = {'rankdir': 'LR', 'label': f"Dataflow for '{gname}'",
                                        'center': 'True'}

                    # keep all direct children and their variables
                    keep = {n for n in G.nodes() if n[lenpre:].count('.') == 0 and
                            n.startswith(prefix)}
                    keep.update({n for n, d in G.nodes(data=True) if 'type_' in d and
                                 n.rpartition('.')[0] in keep})

                    promconns = self._get_prom_conns(group._conn_abs_in2out)

                    for prom_in, (abs_out, abs_ins) in promconns.items():
                        nins = len(abs_ins)
                        the_in = abs_ins[0] if nins == 1 else prom_in
                        if the_in not in G:
                            G.add_node(the_in, type_='input', label=prom_in)
                        else:
                            label = prom_in[:lenpre] if prom_in.startswith(prefix) else prom_in
                            G.nodes[the_in]['label'] = label

                        sysout = prefix + abs_out[lenpre:].partition('.')[0]
                        prom_out = group._var_allprocs_abs2prom['output'][abs_out]

                        if sysout not in G:
                            G.add_node(sysout, **_base_display_map['Group'])

                        label = prom_out[:lenpre] if prom_out.startswith(prefix) else prom_out
                        G.nodes[abs_out]['label'] = label
                        G.add_edge(sysout, abs_out)

                        keep.add(sysout)
                        keep.add(abs_out)

                        for abs_in in abs_ins:
                            sysin = prefix + abs_in[lenpre:].partition('.')[0]
                            if sysin not in G:
                                G.add_node(sysin, **_base_display_map['Group'])
                            if nins == 1:
                                G.add_edge(abs_ins[0], sysin)
                                keep.add(abs_ins[0])
                            else:
                                G.add_edge(prom_in, sysin)
                                keep.add(prom_in)

                        if prom_in in G and nins > 1:
                            G.nodes[prom_in]['fontcolor'] = 'red'
                            G.nodes[prom_in]['tooltip'] = '\n'.join(abs_ins)

                if group.pathname == '':
                    if not recurse:
                        G = nx.subgraph(G, keep)
                else:
                    # we're not the top level group, so get our subgraph of the top level graph
                    ournodes = {n for n in G.nodes() if n.startswith(prefix)}

                    if not recurse:
                        ournodes.update(keep)

                    G = nx.subgraph(G, ournodes)

                if show_boundary and group.pathname:
                    incoming, outgoing = self._get_boundary_conns()
                    G = _add_boundary_nodes(group.pathname, G.copy(), incoming, outgoing, exclude)

                G, node_info = self._decorate_graph_for_display(G, exclude=exclude,
                                                                dvs=desvars, responses=responses)

                if recurse:
                    G = self._apply_clusters(G, node_info)
                else:
                    G = _to_pydot_graph(G)

            elif recurse:
                G = group.compute_sys_graph(comps_only=True, add_edge_info=False)
                if show_boundary and group.pathname:
                    incoming, outgoing = self._get_boundary_conns()
                    # convert var abs names to system abs names
                    incoming = [(in_abs.rpartition('.')[0], out_abs.rpartition('.')[0])
                                for in_abs, out_abs in incoming]
                    outgoing = [(in_abs.rpartition('.')[0], out_abs.rpartition('.')[0])
                                for in_abs, out_abs in outgoing]
                    G = _add_boundary_nodes(group.pathname, G.copy(), incoming, outgoing, exclude)

                G, node_info = self._decorate_graph_for_display(G, exclude=exclude)
                G = self._apply_clusters(G, node_info)
            else:
                G = group.compute_sys_graph(comps_only=False, add_edge_info=False)
                G, _ = self._decorate_graph_for_display(G, exclude=exclude, abs_graph_names=False)
                G = _to_pydot_graph(G)
        else:
            raise ValueError(f"unrecognized graph type '{gtype}'. Allowed types are ['tree', "
                             "'dataflow'].")

        if G is None:
            return

        if outfile is None:
            name = group.pathname.replace('.', '_') if group.pathname else 'model'
            outfile = f"{name}_{gtype}_graph.html"

        return write_graph(G, prog='dot', display=display, outfile=outfile)

    def _get_prom_conns(self, conns):
        """
        Return a dict of promoted connections.

        Parameters
        ----------
        conns : dict
            Dictionary containing absolute connections.

        Returns
        -------
        dict
            Dictionary of promoted connections.
        """
        group = self._group
        abs2prom_in = group._var_allprocs_abs2prom['input']
        prom2abs_in = group._var_allprocs_prom2abs_list['input']
        prefix = group.pathname + '.' if group.pathname else ''
        prom_conns = {}
        for inp, out in conns.items():
            prom = abs2prom_in[inp]
            prom_conns[prom] = (out, [i for i in prom2abs_in[prom] if i.startswith(prefix)])
        return prom_conns

    def _get_graph_display_info(self, display_map=None):
        """
        Return display related metadata for this Group and all of its children.

        Parameters
        ----------
        display_map : dict or None
            A map of classnames to pydot node attributes.

        Returns
        -------
        dict
            Metadata keyed by system pathname.
        """
        group = self._group
        node_info = {}
        for s in group.system_iter(recurse=True, include_self=True):
            meta = s._get_graph_node_meta()
            if display_map and meta['classname'] in display_map:
                meta.update(display_map[meta['classname']])
            elif display_map and meta['base'] in display_map:
                meta.update(display_map[meta['base']])
            else:
                _get_node_display_meta(s, meta)

            ttlist = [f"Name: {s.pathname}"]
            ttlist.append(f"Class: {meta['classname']}")
            if s.linear_solver is not None and not isinstance(s.linear_solver, NonlinearRunOnce):
                ttlist.append(f"Linear Solver: {type(s.linear_solver).__name__}")
            if s.nonlinear_solver is not None and not isinstance(s.nonlinear_solver,
                                                                 NonlinearRunOnce):
                ttlist.append(f"Nonlinear Solver: {type(s.nonlinear_solver).__name__}")
            meta['tooltip'] = '\n'.join(ttlist)
            node_info[s.pathname] = meta.copy()

        if group.comm.size > 1:
            abs2prom = group._var_abs2prom
            all_abs2prom = group._var_allprocs_abs2prom
            if (len(all_abs2prom['input']) != len(abs2prom['input']) or
                    len(all_abs2prom['output']) != len(abs2prom['output'])):
                # not all systems exist in all procs, so must gather info from all procs
                if group._gather_full_data():
                    all_node_info = group.comm.allgather(node_info)
                else:
                    all_node_info = group.comm.allgather({})

                for info in all_node_info:
                    for pathname, meta in info.items():
                        if pathname not in node_info:
                            node_info[pathname] = meta

        return node_info

    def _get_cluster_tree(self, node_info):
        """
        Create a nested collection of pydot Cluster objects to represent the tree of groups.

        Parameters
        ----------
        node_info : dict
            A dict of metadata keyed by pathname.

        Returns
        -------
        pydot.Dot, dict
            The pydot graph and a dict of groups keyed by pathname.
        """
        group = self._group
        pydot_graph = pydot.Dot(graph_type='digraph')
        mypath = group.pathname
        prefix = mypath + '.' if mypath else ''
        groups = {}

        if not mypath:
            groups[''] = pydot.Cluster('', label='Model', style='filled',
                                       fillcolor=_cluster_color(''),
                                       tooltip=node_info['']['tooltip'])
            pydot_graph.add_subgraph(groups[''])

        for varpath in chain(group._var_allprocs_abs2prom['input'],
                             group._var_allprocs_abs2prom['output']):
            group = varpath.rpartition('.')[0].rpartition('.')[0]
            if group not in groups:
                # reverse the list so parents will exist before children
                ancestor_list = list(all_ancestors(group))[::-1]
                for path in ancestor_list:
                    if path.startswith(prefix) or path == mypath:
                        if path not in groups:
                            parent, _, name = path.rpartition('.')
                            groups[path] = pydot.Cluster(path,
                                                         label=path if path == mypath else name,
                                                         tooltip=node_info[path]['tooltip'],
                                                         fillcolor=_cluster_color(path),
                                                         style='filled')
                            if parent and parent.startswith(prefix):
                                groups[parent].add_subgraph(groups[path])
                            elif parent == mypath and parent in groups:
                                groups[parent].add_subgraph(groups[path])
                            else:
                                pydot_graph.add_subgraph(groups[path])

        return pydot_graph, groups

    def _get_tree_graph(self, exclude, display_map=None):
        """
        Create a pydot graph of the system tree (without clusters).

        Parameters
        ----------
        exclude : iter of str
            Iter of pathnames to exclude from the generated graph.
        display_map : dict or None
            A map of classnames to pydot node attributes.

        Returns
        -------
        pydot.Dot
            The pydot tree graph.
        """
        node_info = self._get_graph_display_info(display_map)
        exclude = set(exclude)

        group = self._group
        systems = {}
        pydot_graph = pydot.Dot(graph_type='graph', center=True)
        prefix = group.pathname + '.' if group.pathname else ''
        label = group.name if group.name else 'Model'
        top_node = pydot.Node(label, label=label,
                              **node_info[group.pathname])
        pydot_graph.add_node(top_node)
        systems[group.pathname] = top_node

        for varpath in chain(group._var_allprocs_abs2prom['input'],
                             group._var_allprocs_abs2prom['output']):
            system = varpath.rpartition('.')[0]
            if system not in systems and system not in exclude:
                # reverse the list so parents will exist before children
                ancestor_list = list(all_ancestors(system))[::-1]
                for path in ancestor_list:
                    if path.startswith(prefix):
                        if path not in systems:
                            parent, _, name = path.rpartition('.')
                            kwargs = _filter_meta4dot(node_info[path])
                            systems[path] = pydot.Node(path, label=name, **kwargs)
                            pydot_graph.add_node(systems[path])
                            if parent.startswith(prefix) or parent == group.pathname:
                                pydot_graph.add_edge(pydot.Edge(systems[parent], systems[path]))

        return pydot_graph

    def _decorate_graph_for_display(self, G, exclude=(), abs_graph_names=True, dvs=None,
                                    responses=None):
        """
        Add metadata to the graph for display.

        Returned graph will have any variable nodes containing certain characters relabeled with
        explicit quotes to avoid issues with dot.

        Parameters
        ----------
        G : nx.DiGraph
            The graph to be decorated.
        exclude : iter of str
            Iter of pathnames to exclude from the generated graph.
        abs_graph_names : bool
            If True, use absolute pathnames for nodes in the graph.
        dvs : dict
            Dict of design var metadata keyed on promoted name.
        responses : list of str
            Dict of response var metadata keyed on promoted name.

        Returns
        -------
        nx.DiGraph, dict
            The decorated graph and a dict of node metadata keyed by pathname.
        """
        node_info = self._get_graph_display_info()

        exclude = set(exclude)

        prefix = self._group.pathname + '.' if self._group.pathname else ''

        replace = {}
        for node, meta in G.nodes(data=True):
            if not abs_graph_names:
                node = prefix + node
            if node in node_info:
                meta.update(_filter_meta4dot(node_info[node]))
            if not ('label' in meta and meta['label']):
                meta['label'] = f'"{node.rpartition(".")[2]}"'
            else:
                meta['label'] = f'"{meta["label"]}"'
            if 'type_' in meta:  # variable node
                if node.rpartition('.')[0] in exclude:
                    exclude.add(node)  # remove all variables of excluded components
                # quote node names containing certain characters for use in dot
                if (':' in node or '<' in node) and node not in exclude:
                    replace[node] = f'"{node}"'
                if dvs and node in dvs:
                    meta['shape'] = 'cds'
                elif responses and node in responses:
                    meta['shape'] = 'cds'
                else:
                    meta['shape'] = 'plain'  # just text for variables, otherwise too busy

        if replace:
            G = nx.relabel_nodes(G, replace, copy=True)

        if exclude:
            if not replace:
                G = G.copy()
            G.remove_nodes_from(exclude)

        return G, node_info

    def _apply_clusters(self, G, node_info):
        """
        Group nodes in the graph into clusters.

        Parameters
        ----------
        G : nx.DiGraph
            A pydot graph will be created based on this graph.
        node_info : dict
            A dict of metadata keyed by pathname.

        Returns
        -------
        pydot.Graph
            The corresponding pydot graph with clusters added.
        """
        pydot_graph, groups = self._get_cluster_tree(node_info)
        prefix = self._group.pathname + '.' if self._group.pathname else ''
        boundary_nodes = {'_Incoming', '_Outgoing'}
        pydot_nodes = {}
        for node, meta in G.nodes(data=True):
            noquote_node = node.strip('"')
            kwargs = _filter_meta4dot(meta)
            if 'type_' in meta:  # variable node
                group = noquote_node.rpartition('.')[0].rpartition('.')[0]
            else:
                group = noquote_node.rpartition('.')[0]

            pdnode = pydot_nodes[node] = pydot.Node(node, **kwargs)

            if group and group.startswith(prefix):
                groups[group].add_node(pdnode)
            elif self._group.pathname in groups and node not in boundary_nodes:
                groups[self._group.pathname].add_node(pdnode)
            else:
                pydot_graph.add_node(pdnode)

        for u, v, meta in G.edges(data=True):
            pydot_graph.add_edge(pydot.Edge(pydot_nodes[u], pydot_nodes[v],
                                            **_filter_meta4dot(meta,
                                                               arrowhead='lnormal',
                                                               arrowsize=0.5)))

        # layout graph from left to right
        pydot_graph.set_rankdir('LR')

        return pydot_graph

    def _get_boundary_conns(self):
        """
        Return lists of incoming and outgoing boundary connections.

        Returns
        -------
        tuple
            A tuple of (incoming, outgoing) boundary connections.
        """
        if not self._group.pathname:
            return ([], [])

        top = self._group._problem_meta['model_ref']()
        prefix = self._group.pathname + '.'

        incoming = []
        outgoing = []
        for abs_in, abs_out in top._conn_global_abs_in2out.items():
            if abs_in.startswith(prefix) and not abs_out.startswith(prefix):
                incoming.append((abs_in, abs_out))
            if abs_out.startswith(prefix) and not abs_in.startswith(prefix):
                outgoing.append((abs_in, abs_out))

        return incoming, outgoing


def _get_node_display_meta(s, meta):
    if meta['base'] in _base_display_map:
        meta.update(_base_display_map[meta['base']])
        if s.nonlinear_solver is not None and not isinstance(s.nonlinear_solver, NonlinearRunOnce):
            meta['shape'] = 'box3d'
        elif s.linear_solver is not None and not isinstance(s.linear_solver, LinearRunOnce):
            meta['shape'] = 'box3d'


def write_graph(G, prog='dot', display=True, outfile='graph.html'):
    """
    Write the graph to a file and optionally display it.

    Parameters
    ----------
    G : nx.DiGraph or pydot.Dot
        The graph to be written.
    prog : str
        The graphviz program to use for layout.
    display : bool
        If True, display the graph after writing it.
    outfile : str
        The name of the file to write.  Default is 'graph.html'.

    Returns
    -------
    pydot.Dot
        The graph that was written.
    """
    if pydot is None:
        raise RuntimeError(f"write_graph requires pydot.  Install pydot using "
                           "'pip install pydot'. Note that pydot requires graphviz, which is a "
                           "non-Python application.\nIt can be installed at the system level "
                           "or via a package manager like conda.")

    ext = outfile.rpartition('.')[2]
    if not ext:
        ext = 'html'

    if isinstance(G, nx.Graph):
        pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
    else:
        pydot_graph = G

    ext_conv = {'html': 'svg', 'png': 'png', 'pdf': 'pdf', 'svg': 'svg'}

    try:
        create_func = f"create_{ext_conv[ext]}"
    except KeyError:
        raise AttributeError(f"Can't create a pydot graph file of type '{ext}'.")

    pstr = getattr(pydot_graph, create_func)(prog=prog)

    with open(outfile, 'wb') as f:
        f.write(pstr)

    if display:
        from openmdao.utils.webview import webview
        webview(outfile)

    return pydot_graph


def _to_pydot_graph(G):
    gmeta = G.graph.get('graph', {}).copy()
    gmeta['graph_type'] = 'digraph'
    pydot_graph = pydot.Dot(**gmeta)
    pydot_nodes = {}

    for node, meta in G.nodes(data=True):
        pdnode = pydot_nodes[node] = pydot.Node(node, **_filter_meta4dot(meta))
        pydot_graph.add_node(pdnode)

    for u, v, meta in G.edges(data=True):
        pydot_graph.add_edge(pydot.Edge(pydot_nodes[u], pydot_nodes[v],
                                        **_filter_meta4dot(meta, arrowsize=0.5)))

    # layout graph from left to right
    pydot_graph.set_rankdir('LR')

    return pydot_graph


def _filter_meta4dot(meta, **kwargs):
    """
    Remove unnecessary metadata from the given metadata dict before passing to pydot.

    Parameters
    ----------
    meta : dict
        Metadata dict.
    kwargs : dict
        Additional metadata that will be added only if they are not already present.

    Returns
    -------
    dict
        Metadata dict with unnecessary items removed.
    """
    skip = {'type_', 'local', 'base', 'classname'}
    dct = {k: v for k, v in meta.items() if k not in skip}
    for k, v in kwargs.items():
        if k not in dct:
            dct[k] = v
    return dct


def _add_boundary_nodes(pathname, G, incoming, outgoing, exclude=()):
    """
    Add boundary nodes to the graph.

    Parameters
    ----------
    pathname : str
        Pathname of the current group.
    G : nx.DiGraph
        The graph.
    incoming : list of (str, str)
        List of incoming connections.
    outgoing : list of (str, str)
        List of outgoing connections.
    exclude : list of str
        List of pathnames to exclude from the graph.

    Returns
    -------
    nx.DiGraph
        The modified graph.
    """
    lenpre = len(pathname) + 1 if pathname else 0
    for ex in exclude:
        expre = ex + '.'
        incoming = [(in_abs, out_abs) for in_abs, out_abs in incoming
                    if in_abs != ex and out_abs != ex and
                    not in_abs.startswith(expre) and not out_abs.startswith(expre)]
        outgoing = [(in_abs, out_abs) for in_abs, out_abs in outgoing
                    if in_abs != ex and out_abs != ex and
                    not in_abs.startswith(expre) and not out_abs.startswith(expre)]

    if incoming:
        tooltip = ['External Connections:']
        connstrs = set()
        for in_abs, out_abs in incoming:
            if in_abs in G:
                connstrs.add(f"   {out_abs} -> {in_abs[lenpre:]}")
        tooltip += sorted(connstrs)
        tooltip = '\n'.join(tooltip)
        if connstrs:
            G.add_node('_Incoming', label='Incoming', shape='rarrow', fillcolor='peachpuff3',
                       style='filled', tooltip=f'"{tooltip}"', rank='min')
            for in_abs, out_abs in incoming:
                if in_abs in G:
                    G.add_edge('_Incoming', in_abs, style='dashed', arrowhead='lnormal',
                               arrowsize=0.5)

    if outgoing:
        tooltip = ['External Connections:']
        connstrs = set()
        for in_abs, out_abs in outgoing:
            if out_abs in G:
                connstrs.add(f"   {out_abs[lenpre:]} -> {in_abs}")
        tooltip += sorted(connstrs)
        tooltip = '\n'.join(tooltip)
        G.add_node('_Outgoing', label='Outgoing', shape='rarrow', fillcolor='peachpuff3',
                   style='filled', tooltip=f'"{tooltip}"', rank='max')
        for in_abs, out_abs in outgoing:
            if out_abs in G:
                G.add_edge(out_abs, '_Outgoing', style='dashed', arrowhead='lnormal', arrowsize=0.5)

    return G


def _cluster_color(path):
    """
    Return the color of the cluster that contains the given path.

    The idea here is to make nested clusters stand out wrt their parent cluster.

    Parameters
    ----------
    path : str
        Pathname of a variable.

    Returns
    -------
    int
        The color of the cluster that contains the given path.
    """
    depth = path.count('.') + 1 if path else 0

    ncolors = 10
    maxcolor = 98
    mincolor = 40

    col = maxcolor - (depth % ncolors) * (maxcolor - mincolor) // ncolors
    return f"gray{col}"


def _graph_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao graph' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.description = 'This command requires pydot and graphviz to be installed.'
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-o', action='store', dest='outfile', help='file containing graph output.')
    parser.add_argument('--group', action='store', dest='group', help='pathname of group to graph.')
    parser.add_argument('--type', action='store', dest='type', default='dataflow',
                        help='type of graph (dataflow, tree). Default is dataflow.')
    parser.add_argument('--no-display', action='store_false', dest='show',
                        help="don't display the graph.")
    parser.add_argument('--no-recurse', action='store_false', dest='recurse',
                        help="don't recurse from the specified group down.  This only applies to "
                        "the dataflow graph type.")
    parser.add_argument('--show-vars', action='store_true', dest='show_vars',
                        help="show variables in the graph. This only applies to the dataflow graph."
                        " Default is False.")
    parser.add_argument('--show-boundary', action='store_true', dest='show_boundary',
                        help="show connections to variables outside of the graph. This only "
                        "applies to the dataflow graph. Default is False.")
    parser.add_argument('--autoivc', action='store_true', dest='auto_ivc',
                        help="include the _auto_ivc component in the graph. This applies to "
                             "graphs of the top level group only. Default is False.")


def _graph_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao graph'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _view_graph(problem):
        group = problem.model._get_subsystem(options.group) if options.group else problem.model
        if not options.auto_ivc:
            exclude = {'_auto_ivc'}
        else:
            exclude = set()
        GraphViewer(group).write_graph(gtype=options.type, recurse=options.recurse,
                                       show_vars=options.show_vars, display=options.show,
                                       exclude=exclude, show_boundary=options.show_boundary,
                                       outfile=options.outfile)

    # register the hooks
    hooks._register_hook('final_setup', 'Problem', post=_view_graph, exit=True)
    _load_and_exec(options.file[0], user_args)
