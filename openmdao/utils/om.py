"""
A console script wrapper for multiple openmdao functions.
"""

import sys
import os
import argparse
import importlib.metadata as ilmd

import re
from openmdao import __version__ as version

try:
    import coverage
except ImportError:
    pass
else:
    # the following allows us to do coverage on our command line tools.  It will do nothing
    # unless COVERAGE_PROCESS_START is set in the environment.
    try:
        coverage.process_startup()
    except Exception:
        pass  # in case they're using an old version of coverage


import openmdao.utils.hooks as hooks
from openmdao.visualization.n2_viewer.n2_viewer import _n2_setup_parser, _n2_cmd
from openmdao.visualization.connection_viewer.viewconns import view_connections
from openmdao.visualization.scaling_viewer.scaling_report import _scaling_setup_parser, \
    _scaling_cmd
from openmdao.visualization.timing_viewer.timing_viewer import _timing_setup_parser, _timing_cmd
from openmdao.visualization.dyn_shape_plot import _view_dyn_shapes_setup_parser, \
    _view_dyn_shapes_cmd
try:
    import bokeh
    from openmdao.visualization.meta_model_viewer.meta_model_visualization import view_metamodel
except ImportError:
    bokeh = None
from openmdao.components.meta_model_semi_structured_comp import MetaModelSemiStructuredComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.core.component import Component
from openmdao.devtools.debug import config_summary, tree, comm_info, _dist_conns_cmd, \
    _dist_conns_setup_parser
from openmdao.devtools.itrace import _itrace_exec, _itrace_setup_parser
from openmdao.devtools.iprofile_app.iprofile_app import _iprof_exec, _iprof_setup_parser
from openmdao.devtools.iprofile import _iprof_totals_exec, _iprof_totals_setup_parser
from openmdao.devtools.iprof_mem import _mem_prof_exec, _mem_prof_setup_parser, \
    _mempost_exec, _mempost_setup_parser
from openmdao.error_checking.check_config import _check_config_cmd, _check_config_setup_parser
from openmdao.utils.mpi import MPI
from openmdao.utils.find_cite import print_citations
from openmdao.utils.code_utils import _calltree_setup_parser, _calltree_exec
from openmdao.utils.coloring import _total_coloring_setup_parser, _total_coloring_cmd, \
    _partial_coloring_setup_parser, _partial_coloring_cmd, \
    _view_coloring_setup_parser, _view_coloring_exec
from openmdao.utils.scaffold import _scaffold_setup_parser, _scaffold_exec
from openmdao.utils.file_utils import _load_and_exec, _iter_entry_points
from openmdao.utils.entry_points import _list_installed_setup_parser, _list_installed_cmd, \
    split_ep, _compute_entry_points_setup_parser, _compute_entry_points_exec, \
        _find_repos_setup_parser, _find_repos_exec
from openmdao.utils.reports_system import _list_reports_setup_parser, _list_reports_cmd, \
    _view_reports_setup_parser, _view_reports_cmd
from openmdao.visualization.graph_viewer import _graph_setup_parser, _graph_cmd


def _view_connections_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_connections' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default='connections.html', action='store', dest='outfile',
                        help='html output file.')
    parser.add_argument('-t', '--title', action='store', dest='title',
                        help='title of web page.')
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="don't display in a browser.")
    parser.add_argument('-v', '--show_values', action='store_true', dest='show_values',
                        help="Display values.")
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')


def _view_connections_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao view_connections'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _viewconns(prob):
        if options.title:
            title = options.title
        else:
            title = "Connections for %s" % os.path.basename(options.file[0])
        view_connections(prob, outfile=options.outfile, show_browser=not options.no_browser,
                         show_values=options.show_values, title=title)

    # register the hook
    if options.show_values:
        funcname = 'final_setup'
    else:
        funcname = 'setup'

    def _view_model_w_errors(prob):
        if prob._metadata['saved_errors']:
            # run the viewer here if we've had setup errors. Normally we'd wait until
            # after setup or final_setup.
            _viewconns(prob)
            # errors will result in exit at the end of the _check_collected_errors method

    hooks._register_hook('_check_collected_errors', 'Problem', pre=_view_model_w_errors)
    hooks._register_hook(funcname, class_name='Problem', inst_id=options.problem, post=_viewconns,
                         exit=True)

    _load_and_exec(options.file[0], user_args)


def _meta_model_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao meta_model' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-m', '--metamodel_pathname', action='store', dest='pathname',
                        help='pathname of the metamodel component.')
    parser.add_argument('-r', '--resolution', default=50, type=int,
                        action='store', dest='resolution',
                        help='Number of points to create contour grid')
    parser.add_argument('-p', '--port_number', default=5007, action='store', dest='port_number',
                        help='Port number to open viewer')
    parser.add_argument('--no_browser', action='store_false', dest='browser',
                        help='Bokeh server will start server without browser')


def _meta_model_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao meta_model'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _view_metamodel(prob):
        if bokeh is None:
            print("bokeh must be installed to view a MetaModel.  Use the command:\n",
                  "    pip install bokeh")
            exit()

        hooks._unregister_hook('final_setup', 'Problem')

        mm_types = (MetaModelStructuredComp, MetaModelUnStructuredComp,
                    MetaModelSemiStructuredComp)

        pathname = options.pathname
        port_number = options.port_number
        resolution = options.resolution
        browser = options.browser

        if pathname:
            comp = prob.model._get_subsystem(pathname)
            if comp and isinstance(comp, mm_types):
                view_metamodel(comp, resolution, port_number, browser)
                exit()
        else:
            comp = None

        metamodels = {mm.pathname: mm for
                      mm in prob.model.system_iter(include_self=True, typ=mm_types)}

        mm_names = list(metamodels.keys())
        mm_count = len(mm_names)

        def _mm_list(mm_names, options):
            for mm in mm_names:
                print("openmdao view_mm -m {} {}".format(mm, options.file[0]))

        if mm_count == 0:
            print("No Metamodel components found in model.")

        elif mm_count == 1 and not pathname:
            comp = metamodels[mm_names[0]]
            view_metamodel(comp, resolution, port_number, browser)

        else:
            if not pathname:
                print("\nMetamodel not specified. Try one of the following:\n")
                _mm_list(mm_names, options)
            elif not comp:
                if len(mm_names) > 1:
                    print("\nMetamodel '{}' not found. Try one of the "
                          "following:\n".format(pathname))
                else:
                    print("\nMetamodel '{}' not found. Try the "
                          "following:\n".format(pathname))
                _mm_list(mm_names, options)
            else:
                if len(mm_names) > 1:
                    print("\n'{}' is not a Metamodel. Try one of the "
                          "following:\n".format(pathname))
                else:
                    print("\n'{}' is not a Metamodel. Try the following:\n".format(pathname))
                _mm_list(mm_names, options)

    hooks._register_hook('final_setup', 'Problem', post=_view_metamodel, exit=True)

    _load_and_exec(options.file[0], user_args)


def _config_summary_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao summary' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')


def _config_summary_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao summary'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    hooks._register_hook('final_setup', 'Problem', post=config_summary, exit=True)
    _load_and_exec(options.file[0], user_args)


def _tree_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao tree' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Output file name. By default, output goes to stdout.')
    parser.add_argument('-c', '--colors', action='store_true', dest='show_colors',
                        help="Display colors if the terminal supports it.  Requires 'colorama' "
                             "python package.  Use 'pip install colorama' to install it.")
    parser.add_argument('-d', '--depth', action='store', type=int, dest='depth',
                        default=0, help="Max depth of tree to display.")
    parser.add_argument('-a', '--attr', action='append', default=[], dest='attrs',
                        help='Add an attribute to search for in tree systems.')
    parser.add_argument('-v', '--var', action='append', default=[], dest='vecvars',
                        help='Add a variable to search for in vectors of tree components. '
                             'Use component relative names.')
    parser.add_argument('-r', '--rank', action='store', type=int, dest='rank',
                        default=0, help="Display the tree on this rank (if MPI is active).")
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-s', '--sizes', action='store_true', dest='show_sizes',
                        help="Display input and output sizes.")
    parser.add_argument('--approx', action='store_true', dest='show_approx',
                        help="Show which components compute approximations.")


def _get_tree_filter(attrs, vecvars):
    """
    Pull attributes and input/output vector variables out of a tree System.

    Parameters
    ----------
    attrs : list of str
        Names of attributes (may contain dots).
    vecvars : list of str
        Names of variables contained in the input or output vectors.  Use component relative
        names.

    Returns
    -------
    function
        A function that takes a System and returns a list of name value pairs.
    """
    def _finder(system):
        found = []
        for attr in attrs:
            parts = attr.split('.')  # allow attrs with dots
            try:
                obj = system
                for p in parts:
                    obj = getattr(obj, p)
                found.append((attr, obj))
            except AttributeError:
                pass

        if isinstance(system, Component):
            for var in vecvars:
                if var in system._var_rel2meta:
                    if var in system._outputs:
                        found.append((var, system._outputs[var]))
                    elif var in system._inputs:
                        found.append((var, system._inputs[var]))

        return found

    return _finder


def _tree_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao tree'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if options.outfile is None:
        out = sys.stdout
    else:
        out = open(options.outfile, 'w')

    if options.attrs or options.vecvars:
        filt = _get_tree_filter(options.attrs, options.vecvars)
    else:
        filt = None

    def _tree(prob):
        tree(prob, show_colors=options.show_colors, show_sizes=options.show_sizes,
             show_approx=options.show_approx, filter=filt, max_depth=options.depth,
             rank=options.rank, stream=out)

    # register the hook
    if options.vecvars or options.show_sizes or options.show_approx:
        funcname = 'final_setup'
    else:
        funcname = 'setup'
    hooks._register_hook(funcname, class_name='Problem', inst_id=options.problem, post=_tree,
                         exit=True)
    _load_and_exec(options.file[0], user_args)


def _cite_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao cite' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file.  By default, output goes to stdout.')
    parser.add_argument('-c', '--class', action='append', default=[], dest='classes',
                        help='Find citation for this class.')


def _cite_cmd(options, user_args):
    """
    Run the `openmdao cite` command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if options.outfile is None:
        out = sys.stdout
    else:
        out = open(options.outfile, 'w')

    if not options.classes:
        options.classes = None

    def _cite(prob):
        if not MPI or MPI.COMM_WORLD.rank == 0:
            print_citations(prob, classes=options.classes, out_stream=out)

    hooks._register_hook('setup', 'Problem', post=_cite, exit=True)
    _load_and_exec(options.file[0], user_args)


def _list_pre_post_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao list_pre_post' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file.  By default, output goes to stdout.')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')


def _list_pre_post_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao list_pre_post'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _list_pre_post(prob):
        prob.list_pre_post(outfile=options.outfile)

    # register the hook
    hooks._register_hook('final_setup', class_name='Problem', inst_id=options.problem,
                         post=_list_pre_post, exit=True)

    _load_and_exec(options.file[0], user_args)


def _get_deps(dep_dict: dict, package_name: str) -> None:
    """
    Recursively determine all installed dependency versions and add newly found ones to dep_dict.

    Parameters
    ----------
    dep_dict : dict
        Dictionary with package names as keys and installed versions as values
    package_name : str
        The name of the package to query
    """
    if package_name not in dep_dict:
        try:
            dep_dict[package_name] = ilmd.version(package_name)
            dependencies = ilmd.requires(package_name)
            if dependencies is not None:
                for dep_fullname in dependencies:
                    # requires() returns the full specs, we just want the package name:
                    dep_basename = re.sub(r'^([\w-]+)\W*.*$', r'\1', dep_fullname)
                    _get_deps(dep_dict, dep_basename)

        except ilmd.PackageNotFoundError:
            # Some packages list optional dependencies which are not installed, ignore them
            pass


def _comm_info_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao comm_info' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file.  By default, output goes to stdout.')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help="If True, display comm size and rank range for all Systems. "
                             "Otherwise, display only Systems having a comm size different from "
                             "their parent system.")
    parser.add_argument('--format', action='store', dest='table_format', default='simple_grid',
                        help='Table format.  All formats compatible with the generate_table '
                             'function are available.')


def _comm_info_cmd(options, user_args):
    """
    Run the `openmdao comm_info` command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _comm_info(model):
        if options.problem:
            if model._problem_meta['name'] != options.problem and \
                    model._problem_meta['pathname'] != options.problem:
                return
        elif '/' in model._problem_meta['pathname']:
            # by default, only display comm info for a top level problem
            return

        comm_info(model, outfile=options.outfile, verbose=options.verbose,
                  table_format=options.table_format)

        exit()

    def _set_dyn_hook(prob):
        # set the _comm_info hook to be called right after _setup_procs on the model
        prob.model.pathname = ''
        hooks._register_hook('_setup_procs', class_name='Group', inst_id='', post=_comm_info)
        hooks._setup_hooks(prob.model)

    # register the hook to be called right after setup on the problem
    hooks._register_hook('setup', 'Problem', pre=_set_dyn_hook, ncalls=1)

    _load_and_exec(options.file[0], user_args)


# this dict should contain names mapped to tuples of the form:
#   (setup_parser_func, executor, description)
_command_map = {
    'call_tree': (_calltree_setup_parser, _calltree_exec,
                  "Display the call tree for the specified class method and all 'self' class "
                  "methods it calls."),
    'check': (_check_config_setup_parser, _check_config_cmd,
              'Perform a number of configuration checks on the problem.'),
    'cite': (_cite_setup_parser, _cite_cmd, 'Print citations referenced by the problem.'),
    'comm_info': (_comm_info_setup_parser, _comm_info_cmd,
                  'Print MPI communicator info for systems.'),
    'compute_entry_points': (_compute_entry_points_setup_parser, _compute_entry_points_exec,
                             'Compute entry point declarations to add to the setup.py file.'),
    'dist_conns': (_dist_conns_setup_parser, _dist_conns_cmd,
                   'Display connection information for variables across multiple MPI processes.'),
    'find_repos': (_find_repos_setup_parser, _find_repos_exec,
                   'Find repos on github having openmdao topics.'),
    'graph': (_graph_setup_parser, _graph_cmd, 'Generate a graph for a group.'),
    'iprof': (_iprof_setup_parser, _iprof_exec,
              'Profile calls to particular object instances.'),
    'iprof_totals': (_iprof_totals_setup_parser, _iprof_totals_exec,
                     'Generate total timings of calls to particular object instances.'),
    'list_installed': (_list_installed_setup_parser, _list_installed_cmd,
                       'List installed types recognized by OpenMDAO.'),
    'list_reports': (_list_reports_setup_parser, _list_reports_cmd, 'List available reports.'),
    'mem': (_mem_prof_setup_parser, _mem_prof_exec,
            'Profile memory used by OpenMDAO related functions.'),
    'mempost': (_mempost_setup_parser, _mempost_exec, 'Post-process memory profile output.'),
    'n2': (_n2_setup_parser, _n2_cmd, 'Display an interactive N2 diagram of the problem.'),
    'partial_coloring': (_partial_coloring_setup_parser, _partial_coloring_cmd,
                         'Compute coloring(s) for specified partial jacobians.'),
    'scaffold': (_scaffold_setup_parser, _scaffold_exec,
                 'Generate a simple scaffold for a component.'),
    'scaling': (_scaling_setup_parser, _scaling_cmd, 'View driver scaling report.'),
    'list_pre_post': (_list_pre_post_setup_parser, _list_pre_post_cmd,
                      'Show pre and post setup systems.'),
    'summary': (_config_summary_setup_parser, _config_summary_cmd,
                'Print a short top-level summary of the problem.'),
    'timing': (_timing_setup_parser, _timing_cmd, 'Collect timing information for all systems.'),
    'total_coloring': (_total_coloring_setup_parser, _total_coloring_cmd,
                       'Compute a coloring for the total jacobian.'),
    'trace': (_itrace_setup_parser, _itrace_exec, 'Dump trace output.'),
    'tree': (_tree_setup_parser, _tree_cmd, 'Print the system tree.'),
    'view_coloring': (_view_coloring_setup_parser, _view_coloring_exec, 'View a colored jacobian.'),
    'view_connections': (_view_connections_setup_parser, _view_connections_cmd,
                         'View connections showing values and source/target units.'),
    'view_dyn_shapes': (_view_dyn_shapes_setup_parser, _view_dyn_shapes_cmd,
                        'View the dynamic shape dependency graph.'),
    'view_mm': (_meta_model_parser, _meta_model_cmd, "View a metamodel."),
    'view_reports': (_view_reports_setup_parser, _view_reports_cmd, 'View existing reports.'),
}


def openmdao_cmd():
    """
    Run an 'openmdao' sub-command or list help info for 'openmdao' command or sub-commands.
    """
    # pre-parse sys.argv to split between before and after '--'
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        sys_args = sys.argv[:idx]
        user_args = sys.argv[idx + 1:]
        sys.argv[:] = sys_args
    else:
        user_args = []

    parser = argparse.ArgumentParser(description='OpenMDAO Command Line Tools',
                                     epilog='Use -h after any sub-command for sub-command help, '
                                     'for example, "openmdao tree -h" for help on the "tree" '
                                     'command. If using a tool on a script that takes its own '
                                     'command line arguments, place those arguments after a "--". '
                                     'For example: '
                                     '"openmdao n2 -o foo.html myscript.py -- -x --myarg=bar"')

    ver_group = parser.add_mutually_exclusive_group()
    ver_group.add_argument('--version', action='version', version=version)
    ver_group.add_argument('--dependency_versions', action='store_true', default=False,
                           help="show versions of OpenMDAO and all dependencies, then exit")

    # setting 'dest' here will populate the Namespace with the active subparser name
    subs = parser.add_subparsers(title='Tools', metavar='', dest="subparser_name")
    for p, (parser_setup_func, executor, help_str) in sorted(_command_map.items()):
        subp = subs.add_parser(p, help=help_str)
        parser_setup_func(subp)
        subp.set_defaults(executor=executor)

    # now add any plugin openmdao commands
    epdict = {}
    for ep in _iter_entry_points('openmdao_command'):
        cmd, module, target = split_ep(ep)
        # don't let plugins override the builtin commands
        if cmd in _command_map:
            raise RuntimeError("openmdao plugin command '{}' defined in {} conflicts with "
                               "builtin command '{}'.".format(cmd, module, cmd))
        elif cmd in epdict:
            raise RuntimeError("openmdao plugin command '{}' defined in {} conflicts with a "
                               "another plugin command defined in {}."
                               .format(cmd, module, epdict[cmd][1]))
        epdict[cmd] = (ep, module)

    # sort commands by module and then by command name so commands from plugins will
    # be grouped together.
    for cmd, (ep, module) in sorted(epdict.items(), key=lambda x: x[1][1] + x[0]):
        func = ep.load()
        parser_setup_func, executor, help_str = func()
        pkg = module.split('.', 1)[0]
        subp = subs.add_parser(cmd, help='(%s plugin) ' % pkg + help_str)
        parser_setup_func(subp)
        subp.set_defaults(executor=executor)

    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    cmdargs = [a for a in sys.argv[1:] if a not in ('-h', '--version', '--dependency_versions')]

    # handle case where someone just runs `openmdao <script> [dashed-args]`
    if not set(args).intersection(subs.choices) and len(args) == 1 and os.path.isfile(cmdargs[0]):
        _load_and_exec(args[0], user_args)
    else:
        hooks.use_hooks = True
        # we do a parse_known_args here instead of parse_args so that we can associate errors with
        # the correct subparser.  Otherwise we would just get a top level error message without any
        # sub-command usage info.
        options, unknown = parser.parse_known_args()
        if unknown:
            msg = 'unrecognized arguments: ' + ', '.join(unknown)
            try:
                sub = subs.choices[options.subparser_name]
            except KeyError:
                parser.error(msg)
            else:
                print(sub.format_usage(), file=sys.stderr)
                print(msg, file=sys.stderr)
            parser.exit(2)

        if hasattr(options, 'executor'):
            options.executor(options, user_args)
        elif options.dependency_versions is True:
            dep_versions = {}
            _get_deps(dep_versions, 'openmdao')

            for dep_basename in sorted(dep_versions.keys()):
                print(f'{dep_basename}: {dep_versions[dep_basename]}')
        else:
            print("\nNothing to do.")


if __name__ == '__main__':
    openmdao_cmd()
