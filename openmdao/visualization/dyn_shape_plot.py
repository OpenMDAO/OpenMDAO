"""Functions for plotting the dynamic shapes dependency graph."""

import networkx as nx

from openmdao.core.problem import Problem
from openmdao.utils.mpi import MPI
from openmdao.utils.file_utils import _load_and_exec
import openmdao.utils.hooks as hooks
from openmdao.utils.general_utils import common_subpath
from openmdao.visualization.graph_viewer import write_graph


def _view_dyn_shapes_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_dyn_shapes' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-o', default='shape_dep_graph.png', action='store', dest='outfile',
                        help='plot file.')
    parser.add_argument('-t', '--title', action='store', dest='title', help='title of plot.')
    parser.add_argument('--no_display', action='store_true', dest='no_display',
                        help="don't display the plot.")
    parser.add_argument('--lib', action='store', dest='lib',
                        help="Library to use for plotting, either 'pydot' or 'matplotlib'.")


def _view_dyn_shapes_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao view_dyn_shapes'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _view_shape_graph(model):
        if options.lib not in (None, 'pydot', 'matplotlib'):
            raise RuntimeError(f"Invalid --lib specified for plotting: '{options.lib}'. Valid "
                               "options are 'pydot' or 'matplotlib'.")

        view_dyn_shapes(model, outfile=options.outfile, show=not options.no_display,
                        title=options.title, lib=options.lib)

    def _set_dyn_hook(prob):
        # we can't wait until the end of Problem.setup because we'll die in _setup_sizes
        # if there were any unresolved dynamic shapes, so put the hook immediately after
        # _setup_dynamic_shapes.  inst_id is None here because no system's pathname will
        # have been set at the time this hook is triggered.
        hooks._register_hook('_setup_dynamic_shapes', class_name='Group', inst_id=None,
                             post=_view_shape_graph, exit=True)
        hooks._setup_hooks(prob.model)

    # register the hooks
    hooks._register_hook('setup', 'Problem', pre=_set_dyn_hook, ncalls=1)
    _load_and_exec(options.file[0], user_args)


def view_dyn_shapes(root, outfile='shape_dep_graph.png', show=True, title=None, lib=None):
    """
    Generate a plot file containing the dynamic shape dependency graph.

    Optionally displays the plot.

    Parameters
    ----------
    root : System or Problem
        The top level system or Problem.
    outfile : str, optional
        The name of the plot file.  Defaults to 'shape_dep_graph.png'.
    show : bool, optional
        If True, display the plot. Defaults to True.
    title : str, optional
        Sets the title of the plot.
    lib : str, optional
        Library to use for plotting, either 'pydot' or 'matplotlib'.  If not specified, the
        default is 'pydot' if pydot is installed, otherwise 'matplotlib'.
    """
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    if isinstance(root, Problem):
        system = root.model
    else:
        system = root

    if root.pathname != '':
        raise RuntimeError("view_dyn_shapes cannot be called on a subsystem of the model.  "
                           "Call it with the Problem or the model.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if lib == 'matplotlib':
            raise RuntimeError("matplotlib was specified as the plotting library for "
                               "view_dyn_shapes, but it is not installed.")
        plt = None

    try:
        import pydot
    except ImportError:
        if lib == 'pydot':
            raise RuntimeError("pydot was specified as the plotting library for "
                               "view_dyn_shapes, but it is not installed.")
        pydot = None

    graph = system._shapes_graph

    if graph is None:
        raise RuntimeError("Can't plot dynamic shape dependency graph because it hasn't been "
                           "computed yet.  view_dyn_shapes must be called after problem setup().")

    if graph.order() == 0:
        print("The model has no dynamically shaped variables.")
        return

    if title is None:
        # keep the names from being super long by removing any common subpath
        common = common_subpath(graph.nodes())

        if common:
            title = f"Dynamic shape dependencies in group '{common}'"
            common_idx = len(common) + 1 if common else 0
        else:
            title = "Dynamic shape dependencies"
            common_idx = 0

    abs2meta = system._var_allprocs_abs2meta

    dyn_names = ['shape_by_conn', 'compute_shape', 'copy_shape']

    # label variables with known shape at the start of the algorithm in green, unknowns in red.
    # prepend the shape onto the variable name
    node_colors = []
    node_labels = {}
    for n in graph:
        meta = abs2meta['input'][n] if n in abs2meta['input'] else abs2meta['output'][n]
        shape = meta['shape']
        if shape is None:
            shape = '?'
            node_colors.append('red')
        else:
            for shname in dyn_names:
                if meta.get(shname, False):
                    node_colors.append('blue')
                    break
            else:
                node_colors.append('green')
        node_labels[n] = f"{shape}: {n[common_idx:]}"

    if pydot is not None and (lib is None or lib == 'pydot'):
        Gdot = _to_pydot_graph(graph, node_colors, node_labels)
        write_graph(Gdot, prog='dot', display=show, outfile='dyn_shape_graph.html')
    elif plt is not None:
        nx.draw_networkx(graph, with_labels=True, node_color=node_colors, labels=node_labels)
        plt.axis('off')  # turn of axis
        plt.title(title)
        plt.savefig(outfile)

        if show:
            plt.show()
    else:
        raise RuntimeError("view_dyn_shapes requires either matplotlib or pydot.")

    # TODO: add a legend


def _to_pydot_graph(G, node_colors, node_labels):
    import pydot
    gmeta = G.graph.get('graph', {}).copy()
    gmeta['graph_type'] = 'digraph'
    pydot_graph = pydot.Dot(**gmeta)
    pydot_nodes = {}

    for i, node in enumerate(G.nodes()):
        try:
            label = node_labels[node]
        except TypeError:
            label = node
        color = node_colors[i]
        if color == 'green':
            color = 'lightgreen'
        elif color == 'red':
            color = 'lightcoral'
        elif color == 'blue':
            color = 'lightblue'
        pdnode = pydot_nodes[node] = pydot.Node(node, style='filled', label=label, fillcolor=color)
        pydot_graph.add_node(pdnode)

    for u, v in G.edges():
        pydot_graph.add_edge(pydot.Edge(pydot_nodes[u], pydot_nodes[v]))

    # layout graph from left to right
    # pydot_graph.set_rankdir('LR')

    return pydot_graph
