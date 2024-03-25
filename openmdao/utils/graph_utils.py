"""
Various graph related utilities.
"""
import networkx as nx
from openmdao.utils.file_utils import _load_and_exec
import openmdao.utils.hooks as hooks


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


def write_graph(G, prog='dot', display=True, outfile='graph.svg'):
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
        The name of the file to write.

    Returns
    -------
    pydot.Dot
        The graph that was written.
    """
    from openmdao.utils.webview import webview

    try:
        import pydot
    except ImportError:
        raise RuntimeError("graph requires the pydot package.  You can install it using "
                           "'pip install pydot'.")

    ext = outfile.rpartition('.')[2]
    if not ext:
        ext = 'svg'

    if isinstance(G, nx.Graph):
        pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
    else:
        pydot_graph = G

    try:
        pstr = getattr(pydot_graph, f"create_{ext}")(prog=prog)
    except AttributeError:
        raise AttributeError(f"pydot graph has no 'create_{ext}' method.")

    with open(outfile, 'wb') as f:
        f.write(pstr)

    if display:
        webview(outfile)

    return pydot_graph


def _graph_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao graph' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
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
        group.write_graph(gtype=options.type, recurse=options.recurse,
                          show_vars=options.show_vars, display=options.show, exclude=exclude,
                          show_boundary=options.show_boundary, outfile=options.outfile)

    # register the hooks
    hooks._register_hook('final_setup', 'Problem', post=_view_graph, exit=True)
    _load_and_exec(options.file[0], user_args)

