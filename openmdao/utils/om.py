"""
A console script wrapper for multiple openmdao functions.
"""
from __future__ import print_function

import sys
import os
import argparse
try:
    import pkg_resources
except ImportError:
    pkg_resources = None

from itertools import chain
from six import iteritems

import openmdao.utils.hooks as hooks
from openmdao.visualization.n2_viewer.n2_viewer import n2
from openmdao.visualization.connection_viewer.viewconns import view_connections
from openmdao.visualization.xdsm_viewer.xdsm_writer import write_xdsm, \
    _DEFAULT_BOX_STACKING, _DEFAULT_BOX_WIDTH, _MAX_BOX_LINES, _DEFAULT_OUTPUT_SIDE, _CHAR_SUBS
try:
    import bokeh
    from openmdao.visualization.meta_model_viewer.meta_model_visualization import view_metamodel
except ImportError:
    bokeh = None
from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from openmdao.devtools.debug import config_summary, tree, dump_dist_idxs
from openmdao.devtools.itrace import _itrace_exec, _itrace_setup_parser
from openmdao.devtools.iprofile_app.iprofile_app import _iprof_exec, _iprof_setup_parser
from openmdao.devtools.iprofile import _iprof_totals_exec, _iprof_totals_setup_parser
from openmdao.devtools.iprof_mem import _mem_prof_exec, _mem_prof_setup_parser, \
    _mempost_exec, _mempost_setup_parser
from openmdao.devtools.iprof_utils import _Options
from openmdao.error_checking.check_config import _check_config_cmd, _check_config_setup_parser
from openmdao.utils.mpi import MPI
from openmdao.utils.find_cite import print_citations
from openmdao.utils.code_utils import _calltree_setup_parser, _calltree_exec
from openmdao.utils.coloring import _total_coloring_setup_parser, _total_coloring_cmd, \
    _partial_coloring_setup_parser, _partial_coloring_cmd, \
    _view_coloring_setup_parser, _view_coloring_exec
from openmdao.utils.scaffold import _scaffold_setup_parser, _scaffold_exec
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.file_utils import _load_and_exec
from openmdao.core.component import Component


def _n2_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao n2' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python script or recording containing the model.')
    parser.add_argument('-o', default='n2.html', action='store', dest='outfile',
                        help='html output file.')
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="don't display in a browser.")
    parser.add_argument('--embed', action='store_true', dest='embeddable',
                        help="create embeddable version.")
    parser.add_argument('--title', default=None,
                        action='store', dest='title', help='diagram title.')
    parser.add_argument('--use_declare_partial_info', action='store_true',
                        dest='use_declare_partial_info',
                        help="use declare partial info for internal connectivity.")


def _n2_cmd(options, user_args):
    """
    Process command line args and call n2 on the specified file.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Command line options after '--' (if any).  Passed to user script.
    """
    filename = options.file[0]

    if filename.endswith('.py'):
        # the file is a python script, run as a post_setup hook
        def _viewmod(prob):
            n2(prob, outfile=options.outfile, show_browser=not options.no_browser,
               title=options.title, embeddable=options.embeddable,
               use_declare_partial_info=options.use_declare_partial_info)
            exit()  # could make this command line selectable later

        hooks._register_hook('final_setup', 'Problem', post=_viewmod)

        _load_and_exec(filename, user_args)
    else:
        # assume the file is a recording, run standalone
        n2(filename, outfile=options.outfile, title=options.title,
           show_browser=not options.no_browser, embeddable=options.embeddable,
           use_declare_partial_info=options.use_declare_partial_info)


def _view_model_cmd(options, user_args):
    warn_deprecation("The 'view_model' command has been deprecated. Use 'n2' instead.")
    _n2_cmd(options, user_args)


def _xdsm_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao xdsm' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python script or recording containing the model.')
    parser.add_argument('-o', '--outfile', default='xdsm_out', action='store', dest='outfile',
                        help='XDSM output file. (use pathname without extension)')
    parser.add_argument('-f', '--format', default='html', action='store', dest='format',
                        choices=['html', 'pdf', 'tex'], help='format of XSDM output.')
    parser.add_argument('-m', '--model_path', action='store', dest='model_path',
                        help='Path to system to transcribe to XDSM.')
    parser.add_argument('-r', '--recurse', action='store_true', dest='recurse',
                        help="Don't treat the top level of each name as the source/target "
                             "component.")
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="Don't display in a browser.")
    parser.add_argument('--no_parallel', action='store_true', dest='no_parallel',
                        help="don't show stacked parallel blocks. Only active for 'pdf' and 'tex' "
                             "formats.")
    parser.add_argument('--no_ext', action='store_true', dest='no_extern_outputs',
                        help="Don't show externally connected outputs.")
    parser.add_argument('-s', '--include_solver', action='store_true', dest='include_solver',
                        help="Include the problem model's solver in the XDSM.")
    parser.add_argument('--no_process_conns', action='store_true', dest='no_process_conns',
                        help="Don't add process connections (thin black lines).")
    parser.add_argument('--box_stacking', action='store', default=_DEFAULT_BOX_STACKING,
                        choices=['max_chars', 'vertical', 'horizontal', 'cut_chars', 'empty'],
                        dest='box_stacking', help='Controls the appearance of boxes.')
    parser.add_argument('--box_width', action='store', default=_DEFAULT_BOX_WIDTH,
                        dest='box_width', type=int, help='Controls the width of boxes.')
    parser.add_argument('--box_lines', action='store', default=_MAX_BOX_LINES,
                        dest='box_lines', type=int,
                        help='Limits number of vertical lines in box if box_stacking is vertical.')
    parser.add_argument('--numbered_comps', action='store_true', dest='numbered_comps',
                        help="Display components with numbers.  Only active for 'pdf' and 'tex' "
                        "formats.")
    parser.add_argument('--number_alignment', action='store', dest='number_alignment',
                        choices=['horizontal', 'vertical'], default='horizontal',
                        help='Positions the number either above or in front of the component label '
                        'if numbered_comps is true.')
    parser.add_argument('--output_side', action='store', dest='output_side',
                        default=_DEFAULT_OUTPUT_SIDE,
                        help='Position of the outputs on the diagram. Left or right, or a '
                             'dictionary with component types as keys. Component type key can be '
                             '"optimization", "doe" or "default".')
    parser.add_argument('--legend', action='store_true', dest='legend',
                        help='If True, show legend.')
    parser.add_argument('--class_names', action='store_true', dest='class_names',
                        help='If true, appends class name of the groups/components to the '
                             'component blocks of the diagram.')
    parser.add_argument('--equations', action='store_true', dest='equations',
                        help='If true, for ExecComps their equations are shown in the diagram.')


def _xdsm_cmd(options, user_args):
    """
    Process command line args and call xdsm on the specified file.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Command line options after '--' (if any).  Passed to user script.
    """
    filename = options.file[0]

    kwargs = {}
    for name in ['box_stacking', 'box_width', 'box_lines', 'numbered_comps', 'number_alignment']:
        val = getattr(options, name)
        if val is not None:
            kwargs[name] = val

    if filename.endswith('.py'):
        # the file is a python script, run as a post_setup hook
        def _xdsm(prob):
            write_xdsm(prob, filename=options.outfile, model_path=options.model_path,
                       recurse=options.recurse,
                       include_external_outputs=not options.no_extern_outputs,
                       out_format=options.format,
                       include_solver=options.include_solver, subs=_CHAR_SUBS,
                       show_browser=not options.no_browser, show_parallel=not options.no_parallel,
                       add_process_conns=not options.no_process_conns,
                       output_side=options.output_side,
                       legend=options.legend,
                       class_names=options.class_names,
                       equations=options.equations,
                       **kwargs)
            exit()

        hooks._register_hook('setup', 'Problem', post=_xdsm)

        _load_and_exec(filename, user_args)
    else:
        # assume the file is a recording, run standalone
        write_xdsm(filename, filename=options.outfile, model_path=options.model_path,
                   recurse=options.recurse,
                   include_external_outputs=not options.no_extern_outputs,
                   out_format=options.format,
                   include_solver=options.include_solver, subs=_CHAR_SUBS,
                   show_browser=not options.no_browser, show_parallel=not options.no_parallel,
                   add_process_conns=not options.no_process_conns, output_side=options.output_side,
                   legend=options.legend,
                   class_names=options.class_names,
                   equations=options.equations,
                   **kwargs)


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
        exit()

    # register the hook
    if options.show_values:
        funcname = 'final_setup'
    else:
        funcname = 'setup'
    hooks._register_hook(funcname, class_name='Problem', inst_id=options.problem, post=_viewconns)

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

        mm_types = (MetaModelStructuredComp, MetaModelUnStructuredComp)

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

        if mm_count == 0:
            print("No Metamodel components found in model.")

        elif mm_count == 1 and not pathname:
            comp = metamodels[mm_names[0]]
            view_metamodel(comp, resolution, port_number, browser)

        else:
            try_str = "Try one of the following: {}.".format(mm_names)

            if not pathname:
                print("\nMetamodel not specified. {}".format(try_str))
            elif not comp:
                print("\nMetamodel '{}' not found.\n {}".format(pathname, try_str))
            else:
                print("\n'{}' is not a Metamodel.\n {}".format(pathname, try_str))
        exit()

    hooks._register_hook('final_setup', 'Problem', post=_view_metamodel)

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
    def summary(prob):
        config_summary(prob)
        sys.exit(0)

    hooks._register_hook('final_setup', 'Problem', post=summary)

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
        exit()

    # register the hook
    if options.vecvars or options.show_sizes or options.show_approx:
        funcname = 'final_setup'
    else:
        funcname = 'setup'
    hooks._register_hook(funcname, class_name='Problem', inst_id=options.problem, post=_tree)

    _load_and_exec(options.file[0], user_args)


def _dump_dist_idxs_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao dump_idxs' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file.  By default, output goes to stdout.')
    parser.add_argument('-v', '--vecname', action='store', default='nonlinear', dest='vecname',
                        help='Name of vectors to show indices for.  Default is "nonlinear".')


def _dump_dist_idxs_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao dump_idxs'.

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

    def _dumpdist(prob):
        dump_dist_idxs(prob, vec_name=options.vecname, stream=out)
        exit()

    hooks._register_hook('final_setup', 'Problem', post=_dumpdist)

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

    Returns
    -------
    function
        The hook function.
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
        exit()

    hooks._register_hook('setup', 'Problem', post=_cite)

    _load_and_exec(options.file[0], user_args)


_allowed_types = {
    'component': 'openmdao_components',
    'linear_solver': 'openmdao_lin_solvers',
    'nonlinear_solver': 'openmdao_nl_solvers',
    'line_search': 'openmdao_line_search_solvers',
    'driver': 'openmdao_drivers',
    'case_recorder': 'openmdao_case_recorders',
    'case_reader': 'openmdao_case_readers',
}


def _list_installed_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao list_installed' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('types', nargs='*', help='List these types of installed classes. '
                        'Allowed types are {}.'.format(sorted(_allowed_types)))
    parser.add_argument('-d', '--docs', action='store_true', dest='show_docs',
                        help="Display the class docstrings.")


def _list_installed_cmd(options, user_args):
    """
    Run the `openmdao list_installed` command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.

    Returns
    -------
    function
        The hook function.
    """
    if pkg_resources is None:
        print("You must install pkg_resources in order to use this command.")
        sys.exit(0)

    if not options.types:
        options.types = list(_allowed_types)

    for type_ in options.types:
        if type_ not in _allowed_types:
            raise RuntimeError("Type '{}' is not a valid type.  Try one of {}."
                               .format(type_, sorted(_allowed_types)))
        print("Installed {} types:\n".format(type_))
        epdict = {}
        cwid = 0
        for ep in pkg_resources.iter_entry_points(group=_allowed_types[type_]):
            klass = ep.load()
            epdict[klass.__name__] = (klass.__module__, klass.__doc__)
            if len(klass.__name__) > cwid:
                cwid = len(klass.__name__)

        if epdict:
            print("  {:<{cwid}} {}".format('Class Name', 'Module', cwid=cwid))
            print("  {:<{cwid}} {}".format('----------', '------', cwid=cwid))
        for cname, (mod, doc) in sorted(epdict.items(), key=lambda x: x[1][0] + x[0]):
            line = "  {:<{cwid}} ({})".format(cname, mod, cwid=cwid)
            print(line)
            if options.show_docs and doc:
                print(doc)

        print()


# this dict should contain names mapped to tuples of the form:
#   (setup_parser_func, executor, description)
_command_map = {
    'call_tree': (_calltree_setup_parser, _calltree_exec,
                  "Display the call tree for the specified class method and all 'self' class "
                  "methods it calls."),
    'check': (_check_config_setup_parser, _check_config_cmd,
              'Perform a number of configuration checks on the problem.'),
    'cite': (_cite_setup_parser, _cite_cmd, 'Print citations referenced by the problem'),
    'iprof': (_iprof_setup_parser, _iprof_exec,
              'Profile calls to particular object instances.'),
    'iprof_totals': (_iprof_totals_setup_parser, _iprof_totals_exec,
                     'Generate total timings of calls to particular object instances.'),
    'list_installed': (_list_installed_setup_parser, _list_installed_cmd, 'List installed types.'),
    'mem': (_mem_prof_setup_parser, _mem_prof_exec,
            'Profile memory used by OpenMDAO related functions.'),
    'mempost': (_mempost_setup_parser, _mempost_exec, 'Post-process memory profile output.'),
    'n2': (_n2_setup_parser, _n2_cmd, 'Display an interactive N2 diagram of the problem.'),
    'partial_coloring': (_partial_coloring_setup_parser, _partial_coloring_cmd,
                         'Compute coloring(s) for specified partial jacobians.'),
    'scaffold': (_scaffold_setup_parser, _scaffold_exec,
                 'Generate a simple scaffold for a component.'),
    'summary': (_config_summary_setup_parser, _config_summary_cmd,
                'Print a short top-level summary of the problem.'),
    'total_coloring': (_total_coloring_setup_parser, _total_coloring_cmd,
                       'Compute a coloring for the total jacobian.'),
    'trace': (_itrace_setup_parser, _itrace_exec, 'Dump trace output.'),
    'tree': (_tree_setup_parser, _tree_cmd, 'Print the system tree.'),
    'view_coloring': (_view_coloring_setup_parser, _view_coloring_exec, 'View a colored jacobian.'),
    'view_connections': (_view_connections_setup_parser, _view_connections_cmd,
                         'View connections showing values and source/target units.'),
    'view_mm': (_meta_model_parser, _meta_model_cmd, "View a metamodel."),
    'view_model': (_n2_setup_parser, _view_model_cmd,
                   'Display an interactive N2 diagram of the problem. '
                   '(Deprecated, please use n2 instead.)'),
    'xdsm': (_xdsm_setup_parser, _xdsm_cmd, 'Generate an XDSM diagram of a model.'),
}


# add any dev specific command here that users probably don't want to see
if os.environ.get('OPENMDAO_DEV', '').lower() not in {'0', 'false', 'no', ''}:
    _command_map['dump_idxs'] = (_dump_dist_idxs_setup_parser,
                                 _dump_dist_idxs_cmd,
                                 'Show distributed index information.')


def openmdao_cmd():
    """
    Wrap a number of Problem viewing/debugging command line functions.
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
                                     epilog='Use -h after any sub-command for sub-command help.'
                                     ' If using a tool on a script that takes its own command line'
                                     ' arguments, place those arguments after a "--". For example:'
                                     ' openmdao n2 -o foo.html myscript.py -- -x --myarg=bar')

    # setting 'dest' here will populate the Namespace with the active subparser name
    subs = parser.add_subparsers(title='Tools', metavar='', dest="subparser_name")
    for p, (parser_setup_func, executor, help_str) in sorted(_command_map.items()):
        subp = subs.add_parser(p, help=help_str)
        parser_setup_func(subp)
        subp.set_defaults(executor=executor)

    if pkg_resources is None:
        print("\npkg_resources was not found, so no plugin entry points can be loaded.\n")
    else:
        # now add any plugin openmdao commands
        epdict = {}
        for ep in pkg_resources.iter_entry_points(group='openmdao_commands'):
            p = ep.name
            func = ep.load()
            # don't let plugins override the builtin commands
            if p in _command_map:
                raise RuntimeError("openmdao plugin command '{}' defined in {} conflicts with a "
                                   "builtin command.".format(p, func.__module__))
            elif p in epdict:
                raise RuntimeError("openmdao plugin command '{}' defined in {} conflicts with a "
                                   "another plugin command defined in {}."
                                   .format(p, func.__module__, epdict[p].__module__))
            epdict[p] = func

        for p, func in epdict.items():
            parser_setup_func, executor, help_str = func()
            subp = subs.add_parser(p, help='(plugin) ' + help_str)
            parser_setup_func(subp)
            subp.set_defaults(executor=executor)

    # handle case where someone just runs `openmdao <script> [dashed-args]`
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not set(args).intersection(subs.choices) and len(args) == 1 and os.path.isfile(args[0]):
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
        else:
            print("\nNothing to do.")


if __name__ == '__main__':
    openmdao_cmd()
