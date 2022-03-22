
"""Define a function to view driver timing."""
import os
import sys
import json
import pickle
import atexit
from functools import partial

import openmdao.utils.hooks as hooks
from openmdao.utils.webview import webview
from openmdao.utils.general_utils import default_noraise
from openmdao.utils.mpi import MPI
from openmdao.utils.file_utils import _load_and_exec, _to_filename
import openmdao.visualization.timing_viewer.timer as timer_mod
from openmdao.visualization.timing_viewer.timer import timing_context, _set_timer_setup_hook, \
    _timing_file_iter
from openmdao.utils.om_warnings import issue_warning


_default_timer_methods = sorted(['_solve_nonlinear', '_apply_nonlinear', '_solve_linear',
                                 '_apply_linear', '_linearize'])


def view_timing_text(timing_file, out_stream):
    """
    Print timing data to a file or to stdout.

    Parameters
    ----------
    timing_file : str
        The name of the pickle file contining the timing data.
    out_stream : file-like
        Where the output will be printed.
    """
    for (rank, probname, sysname, _, parallel, method, ncalls,
         avg, min, max, tot) in _timing_file_iter(timing_file):
        parallel = '(parallel)' if parallel else ''
        print(f"{rank:4} (rank) {ncalls:7} (calls) {min:12.6f} (min) "
              f"{max:12.6f} (max) {avg:12.6f} (avg) {tot:12.6f} "
              f"(tot) {parallel} {probname} {sysname}:{method}", file=out_stream)


def view_timing(timing_file, outfile='timing_report.html', show_browser=True, title=''):
    """
    Generate a self-contained html file containing a table of timing data.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    timing_file : str
        The name of the file contining the timing data.
    outfile : str, optional
        The name of the output html file.  Defaults to 'connections.html'.
    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.
    title : str, optional
        Sets the title of the web page.

    Returns
    -------
    dict
        Data to used to generate html file.
    """
    timing_table = []

    idx = 1  # unique ID for use by Tabulator

    # set up timing table data
    for rank, pname, sname, level, parallel, method, ncalls, avgtime, mintime, maxtime, tottime in \
            _timing_file_iter(timing_file):

        dct = {
            'id': idx,
            'rank': rank,
            'probname': pname,
            'sysname': sname,
            'level': level,
            'parallel': parallel,
            'method': method,
            'ncalls': ncalls,
            'avgtime': avgtime,
            'mintime': mintime,
            'maxtime': maxtime,
            'tottime': tottime,
        }

        timing_table.append(dct)

        idx += 1

    data = {
        'title': title,
        'timing_table': timing_table,
    }

    if MPI is None or MPI.COMM_WORLD.rank == 0:

        viewer = 'timing_table.html'

        code_dir = os.path.dirname(os.path.abspath(__file__))
        libs_dir = os.path.join(os.path.dirname(code_dir), 'common', 'libs')
        style_dir = os.path.join(os.path.dirname(code_dir), 'common', 'style')

        with open(os.path.join(code_dir, viewer), "r") as f:
            template = f.read()

        with open(os.path.join(libs_dir, 'tabulator.min.js'), "r") as f:
            tabulator_src = f.read()

        with open(os.path.join(style_dir, 'tabulator.min.css'), "r") as f:
            tabulator_style = f.read()

        with open(os.path.join(libs_dir, 'd3.v6.min.js'), "r") as f:
            d3_src = f.read()

        jsontxt = json.dumps(data, default=default_noraise)

        with open(outfile, 'w') as f:
            s = template.replace("<tabulator_src>", tabulator_src)
            s = s.replace("<tabulator_style>", tabulator_style)
            s = s.replace("<d3_src>", d3_src)
            s = s.replace("<timing_data>", jsontxt)
            f.write(s)

        if show_browser:
            webview(outfile)

    return data


def _show_view(timing_file, options):
    # given a pickled timing file, display based on options.view
    view = options.view.lower()

    if view == 'browser':
        view_timing(timing_file, outfile='timing_report.html', show_browser=True)
    elif view == 'text':
        view_timing_text(timing_file, sys.stdout)
    elif view == 'none':
        pass
    else:
        issue_warning(f"Viewing option '{view}' ignored. Valid options are "
                      "['browser', 'text', 'none'].")


def _postprocess(timing_file, options):
    # this is called by atexit after all timing data has been collected
    timing_managers = timer_mod._timing_managers

    if timing_file is None:
        timing_file = 'timings.pkl'

    if MPI is not None:
        # need to consolidate the timing data from different procs
        all_managers = MPI.COMM_WORLD.gather(timing_managers, root=0)
        if MPI.COMM_WORLD.rank != 0:
            return
    else:
        all_managers = [timing_managers]

    with open(timing_file, 'wb') as f:
        print(f"Saving timing data to '{timing_file}'.")
        pickle.dump(all_managers, f, pickle.HIGHEST_PROTOCOL)

    _show_view(timing_file, options)


def _timing_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao timing' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file where timing data will be stored. By default it '
                        'goes to "timings.pkl".')
    parser.add_argument('-f', '--func', action='append', default=[],
                        dest='funcs', help='Time a specified function. Can be applied multiple '
                        'times to specify multiple functions. '
                        f'Default methods are {_default_timer_methods}.')
    parser.add_argument('-v', '--view', action='store', dest='view', default='browser',
                        help='View the output.  Default view is "browser".  Other options are '
                        '"text" for ascii output or "none" for no output.')
    parser.add_argument('--use_context', action='store_true', dest='use_context',
                        help="If set, timing will only be active within a timing_context.")


def _timing_cmd(options, user_args):
    """
    Implement the 'openmdao timing' command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    filename = _to_filename(options.file[0])
    if filename.endswith('.py'):
        hooks._register_hook('setup', 'Problem', pre=partial(_set_timer_setup_hook, options))

        if options.outfile is not None and MPI:
            outfile = f"{options.outfile}.{MPI.COMM_WORLD.rank}"
        else:
            outfile = options.outfile

        if not options.funcs:
            options.funcs = _default_timer_methods.copy()

        # register an atexit function to write out all of the timing data
        atexit.register(partial(_postprocess, outfile, options))

        with timing_context(not options.use_context):
            _load_and_exec(options.file[0], user_args)

    else:  # assume file is a pickle file

        _show_view(options.file[0], options)
