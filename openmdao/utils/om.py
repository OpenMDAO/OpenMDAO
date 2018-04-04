"""
A console script wrapper for multiple openmdao functions.
"""
from __future__ import print_function

import sys
import os
import argparse
from six import iteritems

from openmdao.core.problem import Problem
from openmdao.utils.coloring import _simul_coloring_setup_parser, _simul_coloring_cmd
from openmdao.devtools.problem_viewer.problem_viewer import view_model
from openmdao.devtools.viewconns import view_connections
from openmdao.devtools.debug import config_summary, tree, dump_dist_idxs
from openmdao.devtools.itrace import _itrace_exec, _itrace_setup_parser
from openmdao.devtools.iprofile_app.iprofile_app import _iprof_exec, _iprof_setup_parser
from openmdao.devtools.iprofile import _iprof_totals_exec, _iprof_totals_setup_parser
from openmdao.devtools.iprof_mem import _mem_prof_exec, _mem_prof_setup_parser
from openmdao.error_checking.check_config import _check_config_cmd, _check_config_setup_parser
from openmdao.devtools.iprof_utils import _Options
from openmdao.utils.mpi import MPI
from openmdao.utils.find_cite import print_citations


def _view_model_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_model' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default='n2.html', action='store', dest='outfile',
                        help='html output file.')
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="don't display in a browser.")
    parser.add_argument('--embed', action='store_true', dest='embeddable',
                        help="create embeddable version.")
    parser.add_argument('--draw_potential_connections', action='store_true',
                        dest='draw_potential_connections',
                        help="draw potential connections.")


def _view_model_cmd(options):
    """
    Return the post_setup hook function for 'openmdao view_model'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    def _viewmod(prob):
        view_model(prob, outfile=options.outfile,
                   show_browser=not options.no_browser,
                   embeddable=options.embeddable,
                   draw_potential_connections=options.draw_potential_connections)
        exit()  # could make this command line selectable later
    return _viewmod


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
        view_connections(prob, outfile=options.outfile, show_browser=not options.no_browser)
        exit()
    return _viewconns


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
#   (setup_parser_func, func)
_post_setup_map = {
    'view_model': (_view_model_setup_parser, _view_model_cmd),
    'view_connections': (_view_connections_setup_parser, _view_connections_cmd),
    'summary': (_config_summary_setup_parser, _config_summary_cmd),
    'tree': (_tree_setup_parser, _tree_cmd),
    'dump_idxs': (_dump_dist_idxs_setup_parser, _dump_dist_idxs_cmd),
    'simul_coloring': (_simul_coloring_setup_parser, _simul_coloring_cmd),
    'cite': (_cite_setup_parser, _cite_cmd),
    'check': (_check_config_setup_parser, _check_config_cmd),
}


# Other non-post-setup functions go here
_iprof_map = {
    'trace': (_itrace_setup_parser, _itrace_exec),
    'iprof': (_iprof_setup_parser, _iprof_exec),
    'iprof_totals': (_iprof_totals_setup_parser, _iprof_totals_exec),
    'mem': (_mem_prof_setup_parser, _mem_prof_exec),
}


def openmdao_cmd():
    """
    Wrap a number of Problem viewing/debugging command line functions.
    """
    parser = argparse.ArgumentParser()

    subs = parser.add_subparsers(help='sub-command help')
    for p, (parser_setup_func, cmd) in iteritems(_post_setup_map):
        subp = subs.add_parser(p)
        parser_setup_func(subp)
        subp.set_defaults(func=cmd, executor=_post_setup_exec)

    for p, (parser_setup_func, cmd) in iteritems(_iprof_map):
        subp = subs.add_parser(p)
        parser_setup_func(subp)
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
