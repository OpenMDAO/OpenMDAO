"""Functions for plotting the dynamic units dependency graph."""

import networkx as nx

from openmdao.core.problem import Problem
from openmdao.utils.mpi import MPI
from openmdao.utils.file_utils import _load_and_exec
import openmdao.utils.hooks as hooks
from openmdao.utils.general_utils import common_subpath


def _view_dyn_units_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_dyn_units' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-o', default='unit_dep_graph.png', action='store', dest='outfile',
                        help='plot file.')
    parser.add_argument('-t', '--title', action='store', dest='title', help='title of plot.')
    parser.add_argument('--no_display', action='store_true', dest='no_display',
                        help="don't display the plot.")


def _view_dyn_units_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao view_dyn_units'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _view_unit_graph(model):
        view_dyn_units(model, outfile=options.outfile, show=not options.no_display,
                       title=options.title)

    def _set_dyn_hook(prob):
        # we can't wait until the end of Problem.setup because we'll die in _setup_sizes
        # if there were any unresolved dynamic shapes, so put the hook immediately after
        # _setup_dynamic_properties.  inst_id is None here because no system's pathname will
        # have been set at the time this hook is triggered.
        hooks._register_hook('_setup_dynamic_properties', class_name='Group', inst_id=None,
                             post=_view_unit_graph, exit=True)
        hooks._setup_hooks(prob.model)

    # register the hooks
    hooks._register_hook('setup', 'Problem', pre=_set_dyn_hook, ncalls=1)
    _load_and_exec(options.file[0], user_args)


def view_dyn_units(root, outfile='unit_dep_graph.png', show=True, title=None):
    """
    Generate a plot file containing the dynamic unit dependency graph.

    Optionally displays the plot.

    Parameters
    ----------
    root : System or Problem
        The top level system or Problem.

    outfile : str, optional
        The name of the plot file.  Defaults to 'unit_dep_graph.png'.

    show : bool, optional
        If True, display the plot. Defaults to True.

    title : str, optional
        Sets the title of the plot.
    """
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    if isinstance(root, Problem):
        system = root.model
    else:
        system = root

    if root.pathname != '':
        raise RuntimeError("view_dyn_units cannot be called on a subsystem of the model.  "
                           "Call it with the Problem or the model.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("The view_dyn_units command requires matplotlib.")

    graph = system._units_graph

    if graph is None:
        raise RuntimeError("Can't plot dynamic unit dependency graph because it hasn't been "
                           "computed yet.  view_dyn_units must be called after problem setup().")

    if graph.order() == 0:
        print("The model has no dynamically unitd variables.")
        return

    if title is None:
        # keep the names from being super long by removing any common subpath
        common = common_subpath(graph.nodes())

        if common:
            title = f"Dynamic unit dependencies in group '{common}'"
            common_idx = len(common) + 1 if common else 0
        else:
            title = "Dynamic unit dependencies"
            common_idx = 0

    abs2meta = system._var_allprocs_abs2meta

    dyn_names = ['unit_by_conn', 'compute_unit', 'copy_unit']

    # label variables with known unit at the start of the algorithm in green, unknowns in red.
    # prepend the unit onto the variable name
    node_colors = []
    node_labels = {}
    for n in graph:
        try:
            meta = abs2meta['input'][n] if n in abs2meta['input'] else abs2meta['output'][n]
            units = meta['units']
        except KeyError:
            units = None

        for shname in dyn_names:
            if meta.get(shname, False):
                is_dynamic = True
                break
        else:
            is_dynamic = False

        if is_dynamic:
            if units is None:
                node_colors.append('red')
                units = '?'
            else:
                node_colors.append('blue')
        else:
            if units is None:
                units = 'None'
            node_colors.append('green')

        node_labels[n] = f"{units}: {n[common_idx:]}"

    nx.draw_networkx(graph, with_labels=True, node_color=node_colors, labels=node_labels)
    plt.axis('off')  # turn of axis
    plt.title(title)
    plt.savefig(outfile)

    if show:
        plt.show()

    # TODO: add a legend
    # TODO: use a better graph plotting lib, maybe D3 or something else, to get better layout
