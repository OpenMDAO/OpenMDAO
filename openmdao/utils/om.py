"""
A console script wrapper for multiple openmdao functions.
"""
from __future__ import print_function

import sys
import os
import argparse
from itertools import chain
from six import iteritems

from openmdao.core.problem import Problem
from openmdao.visualization.n2_viewer.n2_viewer import n2
from openmdao.visualization.connection_viewer.viewconns import view_connections
from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
try:
    import bokeh
    from openmdao.visualization.meta_model_viewer.meta_model_visualization import view_metamodel
except ImportError:
    bokeh = None
from openmdao.devtools.debug import config_summary, tree, dump_dist_idxs
from openmdao.devtools.itrace import _itrace_exec, _itrace_setup_parser
from openmdao.devtools.iprofile_app.iprofile_app import _iprof_exec, _iprof_setup_parser
from openmdao.devtools.iprofile import _iprof_totals_exec, _iprof_totals_setup_parser
from openmdao.devtools.iprof_mem import _mem_prof_exec, _mem_prof_setup_parser, \
    _mempost_exec, _mempost_setup_parser
from openmdao.devtools.iprof_utils import _Options
from openmdao.visualization.xdsm_viewer.xdsm_writer import write_xdsm, \
    _DEFAULT_BOX_STACKING, _DEFAULT_BOX_WIDTH, _MAX_BOX_LINES, _DEFAULT_OUTPUT_SIDE, _CHAR_SUBS
from openmdao.error_checking.check_config import _check_config_cmd, _check_config_setup_parser
from openmdao.utils.mpi import MPI
from openmdao.utils.find_cite import print_citations
from openmdao.utils.code_utils import _calltree_setup_parser, _calltree_exec
from openmdao.utils.coloring import _total_coloring_setup_parser, _total_coloring_cmd, \
    _sparsity_setup_parser, _sparsity_cmd, _partial_coloring_setup_parser, _partial_coloring_cmd, \
    _view_coloring_setup_parser, _view_coloring_exec
from openmdao.utils.scaffold import _scaffold_setup_parser, _scaffold_exec
from openmdao.utils.general_utils import warn_deprecation


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


def _n2_cmd(options):
    """
    Process command line args and call n2 on the specified file.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    """
    filename = options.file[0]

    if filename.endswith('.py'):
        # the file is a python script, run as a post_setup hook
        def _viewmod(prob):
            n2(prob, outfile=options.outfile, show_browser=not options.no_browser,
               title=options.title, embeddable=options.embeddable,
               use_declare_partial_info=options.use_declare_partial_info)
            exit()  # could make this command line selectable later

        options.func = lambda options: _viewmod
        _post_setup_exec(options)
    else:
        # assume the file is a recording, run standalone
        n2(filename, outfile=options.outfile, title=options.title,
           show_browser=not options.no_browser, embeddable=options.embeddable,
           use_declare_partial_info=options.use_declare_partial_info)


def _view_model_cmd(options):
    warn_deprecation("The 'view_model' command has been deprecated. Use 'n2' instead.")
    _n2_cmd(options)


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
                             'component blocks of the diagram..')


def _xdsm_cmd(options):
    """
    Process command line args and call xdsm on the specified file.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
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
                       **kwargs)
            exit()

        options.func = lambda options: _xdsm

        _post_setup_exec(options)
    else:
        # assume the file is a recording, run standalone
        write_xdsm(filename, filename=options.outfile, model_path=options.model_path,
                   recurse=options.recurse,
                   include_external_outputs=not options.no_extern_outputs,
                   out_format=options.format,
                   include_solver=options.include_solver, subs=_CHAR_SUBS,
                   show_browser=not options.no_browser, show_parallel=not options.no_parallel,
                   add_process_conns=not options.no_process_conns, output_side=options.output_side,
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


def _view_connections_cmd(options):
    """
    Return the post_setup hook function for 'openmdao view_connections'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    def _viewconns(prob):
        if options.title:
            title = options.title
        else:
            title = "Connections for %s" % os.path.basename(options.file[0])
        view_connections(prob, outfile=options.outfile, show_browser=not options.no_browser,
                         title=title)
        exit()
    return _viewconns


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
    parser.add_argument('-r', '--resolution', default=50, action='store', dest='resolution',
                        help='Number of points to create contour grid')
    parser.add_argument('-p', '--port_number', default=5007, action='store', dest='port_number',
                        help='Port number to open viewer')


def _meta_model_cmd(options):
    """
    Return the post_setup hook function for 'openmdao meta_model'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    def _view_metamodel(prob):
        if bokeh is None:
            print("bokeh must be installed to view a MetaModel.  Use the command:\n",
                  "    pip install bokeh")
            exit()

        Problem._post_setup_func = None

        mm_types = (MetaModelStructuredComp, MetaModelUnStructuredComp)

        pathname = options.pathname
        port_number = options.port_number
        try:
            resolution = int(options.resolution)
        except ValueError:
            print("Resolution must be an int")
            exit()

        if pathname:
            comp = prob.model._get_subsystem(pathname)
            if comp and isinstance(comp, mm_types):
                view_metamodel(comp, resolution, port_number)
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
            view_metamodel(comp, resolution, port_number)

        else:
            try_str = "Try one of the following: {}.".format(mm_names)

            if not pathname:
                print("\nMetamodel not specified. {}".format(try_str))
            elif not comp:
                print("\nMetamodel '{}' not found.\n {}".format(pathname, try_str))
            else:
                print("\n'{}' is not a Metamodel.\n {}".format(pathname, try_str))
        exit()

    return _view_metamodel


def _config_summary_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao summary' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')


def _config_summary_cmd(options):
    """
    Return the post_setup hook function for 'openmdao summary'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    def summary(prob):
        config_summary(prob)
        exit()
    return summary


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
                        help='Add a variable to search for in vectors of tree systems.')
    parser.add_argument('-r', '--rank', action='store', type=int, dest='rank',
                        default=0, help="Display the tree on this rank (if MPI is active).")


def _get_tree_filter(attrs, vecvars):
    """
    Pull attributes and input/output vector variables out of a tree System.

    Parameters
    ----------
    attrs : list of str
        Names of attributes (may contain dots).
    vecvars : list of str
        Names of variables contained in the input or output vectors.

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

        for var in vecvars:
            if var in system._outputs:
                found.append((var, system._outputs[var]))
            elif var in system._inputs:
                found.append((var, system._inputs[var]))

        return found

    return _finder


def _tree_cmd(options):
    """
    Return the post_setup hook function for 'openmdao tree'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
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
        tree(prob, show_colors=options.show_colors,
             filter=filt, max_depth=options.depth, rank=options.rank, stream=out)
        exit()
    return _tree


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


def _dump_dist_idxs_cmd(options):
    """
    Return the post_setup hook function for 'openmdao dump_idxs'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    if options.outfile is None:
        out = sys.stdout
    else:
        out = open(options.outfile, 'w')

    def _dumpdist(prob):
        dump_dist_idxs(prob, vec_name=options.vecname, stream=out)
        exit()
    return _dumpdist


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


def _cite_cmd(options):
    """
    Return the post setup hook function for `openmdao cite`.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
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

    return _cite


def _post_setup_exec(options):
    """
    Use this as executor for commands that run as Problem post-setup commands.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    """
    progname = options.file[0]

    sys.path.insert(0, os.path.dirname(progname))

    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')

    globals_dict = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    if options.func is not None:
        Problem._post_setup_func = options.func(options)

    exec(code, globals_dict)


# NOTE: any post_setup functions must handle their own exit behavior. If you want them
# to exit after running, exit() must be called from within your function.  This also gives
# you the option of controlling the exit behavior via a command line argument.

# All post-setup functions go here.
# this dict should contain names mapped to tuples of the form:
#   (setup_parser_func, func, description)
_post_setup_map = {
    'view_connections': (_view_connections_setup_parser, _view_connections_cmd,
                         'Connection viewer showing values and source/target units.'),
    'summary': (_config_summary_setup_parser, _config_summary_cmd,
                'Print a short top-level summary of the problem.'),
    'tree': (_tree_setup_parser, _tree_cmd, 'Print the system tree.'),
    'dump_idxs': (_dump_dist_idxs_setup_parser, _dump_dist_idxs_cmd,
                  'Show distributed index information.'),
    'total_coloring': (_total_coloring_setup_parser, _total_coloring_cmd,
                       'Compute a coloring for the total jacobian.'),
    'partial_coloring': (_partial_coloring_setup_parser, _partial_coloring_cmd,
                         'Compute coloring(s) for specified partial jacobians.'),
    'total_sparsity': (_sparsity_setup_parser, _sparsity_cmd,
                       'Compute the sparsity pattern of the total jacobian.'),
    'cite': (_cite_setup_parser, _cite_cmd,
             'Print citations referenced by problem'),
    'check': (_check_config_setup_parser, _check_config_cmd,
              'Perform a number of configuration checks on the problem.'),
    'view_mm': (_meta_model_parser, _meta_model_cmd, "Meta Model Viewer.")
}


# Other non-post-setup functions go here
_non_post_setup_map = {
    'n2': (_n2_setup_parser, _n2_cmd, 'Display an interactive N2 diagram of the problem.'),
    'view_model': (_n2_setup_parser, _view_model_cmd,
                   'Display an interactive N2 diagram of the problem. '
                   '(Deprecated, please use n2 instead.)'),
    'trace': (_itrace_setup_parser, _itrace_exec, 'Dump trace output.'),
    'call_tree': (_calltree_setup_parser, _calltree_exec,
                  "Display the call tree for the specified class method and all 'self' class "
                  "methods it calls."),
    'iprof': (_iprof_setup_parser, _iprof_exec,
              'Profiling of calls to particular object instances.'),
    'iprof_totals': (_iprof_totals_setup_parser, _iprof_totals_exec,
                     'Total timings of calls to particular object instances.'),
    'mem': (_mem_prof_setup_parser, _mem_prof_exec, 'Memory profiler.'),
    'mempost': (_mempost_setup_parser, _mempost_exec, 'Post-processor for memory profile output.'),
    'view_coloring': (_view_coloring_setup_parser, _view_coloring_exec,
                      'Colored jacobian viewer.'),
    'xdsm': (_xdsm_setup_parser, _xdsm_cmd, 'XDSM viewer.'),
    'scaffold': (_scaffold_setup_parser, _scaffold_exec,
                 'Generate a simple scaffold for a component.'),
}


def openmdao_cmd():
    """
    Wrap a number of Problem viewing/debugging command line functions.
    """
    parser = argparse.ArgumentParser(description='OpenMDAO Command Line Tools',
                                     epilog='Use -h after any sub-command for sub-command help.')

    subs = parser.add_subparsers(title='Tools', metavar='')
    for p, (parser_setup_func, cmd, help_str) in sorted(chain(_post_setup_map.items(),
                                                              _non_post_setup_map.items())):
        subp = subs.add_parser(p, help=help_str)
        parser_setup_func(subp)
        if p in _post_setup_map:
            subp.set_defaults(func=cmd, executor=_post_setup_exec)
        else:
            subp.set_defaults(executor=cmd)

    # handle case where someone just runs `openmdao <script>`
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not set(args).intersection(subs.choices) and len(args) == 1 and os.path.isfile(args[0]):
        _post_setup_exec(_Options(file=[args[0]], func=None))
    else:
        options = parser.parse_args()
        if hasattr(options, 'executor'):
            options.executor(options)
        else:
            print("\nNothing to do.")


if __name__ == '__main__':
    openmdao_cmd()
