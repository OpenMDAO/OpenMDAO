
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
    _timing_file_iter, _get_par_child_info
from openmdao.utils.om_warnings import issue_warning
from openmdao.core.constants import _DEFAULT_OUT_STREAM


_default_timer_methods = sorted(['_solve_nonlinear'])


def view_timing_dump(timing_file, out_stream=_DEFAULT_OUT_STREAM):
    """
    Print a timing dump to a file or to stdout.

    Parameters
    ----------
    timing_file : str
        The name of the pickle file contining the timing data.
    out_stream : file-like or None
        Where the output will be printed. If None, generate no output.
    """
    if out_stream is None:
        return
    elif out_stream is _DEFAULT_OUT_STREAM:
        out_stream = sys.stdout

    for (rank, probname, classname, sysname, _, parallel, nprocs, method, ncalls,
         avg, min, max, tot, global_tot) in _timing_file_iter(timing_file):
        parallel = '(parallel)' if parallel else ''
        pct = tot / global_tot * 100.
        print(f"{rank:4} (rank) {nprocs:4} (nprocs) {ncalls:7} (calls) {min:12.6f} (min) "
              f"{max:12.6f} (max) {avg:12.6f} (avg) {tot:12.6f} "
              f"(tot) {pct:6.2f} % {parallel} {probname} {sysname}:{method}", file=out_stream)


def view_MPI_timing(timing_file, method='_solve_nonlinear', out_stream=_DEFAULT_OUT_STREAM):
    """
    Print timings of direct children of ParallelGroups to a file or to stdout.

    Parameters
    ----------
    timing_file : str
        The name of the pickle file contining the timing data.
    method : str
        Name of method to show timings for. Default is _solve_nonlinear.
    out_stream : file-like or None
        Where the output will be printed. If None, generate no output.

    Returns
    -------
    dict or None
        Timing info dict or None.
    """
    if out_stream is None:
        return
    elif out_stream is _DEFAULT_OUT_STREAM:
        out_stream = sys.stdout

    seen = set()
    parinfo = _get_par_child_info(_timing_file_iter(timing_file), method)

    if not parinfo:
        return

    cols = ['System', 'Rank', 'Calls', 'Avg Time', 'Min Time', 'Max_time', 'Total Time']
    colspc = ['-' * len(s) for s in cols]
    for key, sdict in parinfo.items():
        probname, parentsys = key
        if probname not in seen:
            print(f"\n\nProblem: {probname}   method: {method}", file=out_stream)
            seen.add(probname)

        slist = list(sdict.items())
        # get ncalls from a subsystem (ncalls will be same for all and for the parent)
        ncalls = slist[0][1][0][1]

        print(f"\n  Parallel group: {parentsys} (ncalls = {ncalls})\n", file=out_stream)
        print(f"  {cols[0]:20}  {cols[1]:>5}  {cols[3]:>12} {cols[4]:>12} "
              f"{cols[5]:>12} {cols[6]:>12}", file=out_stream)
        print(f"  {colspc[0]:20}  {colspc[1]:>5}  {colspc[3]:>12} {colspc[4]:>12} "
              f"{colspc[5]:>12} {colspc[6]:>12}", file=out_stream)
        for sysname, dlist in slist:
            relname = sysname.rpartition('.')[2]
            for (rank, ncalls, avg, tmin, tmax, ttot) in dlist:
                print(f"  {relname:20}  {rank:>5}  {avg:12.4f} {tmin:12.4f}"
                      f" {tmax:12.4f} {ttot:12.4f}", file=out_stream)

    return parinfo


def view_timing(timing_file, outfile='timing_report.html', show_browser=True):
    """
    Generate a self-contained html file containing an interactive table of timing data.

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

    Returns
    -------
    dict
        Data to used to generate html file.
    """
    timing_table = []

    tot_by_rank = {}
    max_rank = 0

    parent_dict = {}  # used for computing pct of parent timing

    # set up timing table data
    for rank, probname, cname, sname, level, parallel, nprocs, method, ncalls, avgtime, mintime, \
            maxtime, tottime, globaltot in _timing_file_iter(timing_file):

        if rank > max_rank:
            max_rank = rank

        dct = {
            'id': len(timing_table) + 1,  # unique ID for use by Tabulator
            'rank': rank,
            'probname': probname,
            'classname': cname,
            'sysname': sname,
            'level': level,
            'parallel': parallel,
            'nprocs': nprocs,
            'method': method,
            'ncalls': ncalls,
            'avgtime': avgtime,
            'mintime': mintime,
            'maxtime': maxtime,
            'tottime': tottime,
            'pct': tottime / globaltot * 100.,
        }

        timing_table.append(dct)
        tot_by_rank[rank] = globaltot

        parent_dict[(rank, probname, sname, method)] = dct

    data = {
        'title': f"Total time: {max(tot_by_rank.values() if tot_by_rank else [0.]):12.6f} sec",
        'timing_table': timing_table,
        'tot_procs': max_rank + 1,
    }

    if MPI is None or MPI.COMM_WORLD.rank == 0:

        viewer = 'timing_table.html'

        code_dir = os.path.dirname(os.path.abspath(__file__))
        libs_dir = os.path.join(os.path.dirname(code_dir), 'common', 'libs')
        style_dir = os.path.join(os.path.dirname(code_dir), 'common', 'style')

        with open(os.path.join(code_dir, viewer), "r", encoding='utf-8') as f:
            template = f.read()

        with open(os.path.join(libs_dir, 'tabulator.5.4.4.min.js'), "r", encoding='utf-8') as f:
            tabulator_src = f.read()

        with open(os.path.join(style_dir, 'tabulator.5.4.4.min.css'), "r", encoding='utf-8') as f:
            tabulator_style = f.read()

        with open(os.path.join(libs_dir, 'd3.v6.min.js'), "r", encoding='utf-8') as f:
            d3_src = f.read()

        jsontxt = json.dumps(data, default=default_noraise)

        with open(outfile, 'w', encoding='utf-8') as f:
            s = template.replace("<tabulator_src>", tabulator_src)
            s = s.replace("<tabulator_style>", tabulator_style)
            s = s.replace("<d3_src>", d3_src)
            s = s.replace("<timing_data>", jsontxt)
            f.write(s)

        if show_browser:
            webview(outfile)

    return data


_view_options = [
    'text',
    'browser',
    'no_browser',
    'dump',
    'none'
]


def _show_view(timing_file, options):
    # given a timing file, display based on options.view
    view = options.view.lower()

    if view == 'text':
        for f in options.funcs:
            ret = view_MPI_timing(timing_file, method=f, out_stream=sys.stdout)
            if ret is None:
                issue_warning(f"Could find no children of a ParallelGroup running method '{f}'.")
    elif view == 'browser' or view == 'no_browser':
        view_timing(timing_file, outfile='timing_report.html', show_browser=view == 'browser')
    elif view == 'dump':
        view_timing_dump(timing_file, out_stream=sys.stdout)
    elif view == 'none':
        pass
    else:
        issue_warning(f"Viewing option '{view}' ignored. Valid options are "
                      f"{_view_options}.")


def _postprocess(options):
    # this is called by atexit after all timing data has been collected
    # Note that this will not be called if the program exits via sys.exit() with a nonzero
    # exit code.
    timing_managers = timer_mod._timing_managers
    timing_file = options.outfile

    if timing_file is None:
        timing_file = 'timings.pkl'

    timing_data = (timing_managers, timer_mod._total_time)

    if MPI is not None:
        # need to consolidate the timing data from different procs
        all_managers = MPI.COMM_WORLD.gather(timing_data, root=0)
        if MPI.COMM_WORLD.rank != 0:
            return
    else:
        all_managers = [timing_data]

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
    parser.add_argument('file', nargs=1, help='Python file containing the model, or pickle file '
                        'containing previously recorded timing data.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file where timing data will be stored. By default it '
                        'goes to "timings.pkl".')
    parser.add_argument('-f', '--func', action='append', default=[],
                        dest='funcs', help='Time a specified function. Can be applied multiple '
                        'times to specify multiple functions. '
                        f'Default methods are {_default_timer_methods}.')
    parser.add_argument('-v', '--view', action='store', dest='view', default='text',
                        help="View of the output.  Default view is 'text', which shows timing for "
                        "each direct child of a parallel group across all ranks. Other options are "
                        f"{_view_options[1:]}.")
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
    if not options.funcs:
        options.funcs = _default_timer_methods.copy()

    filename = _to_filename(options.file[0])
    if filename.endswith('.py'):
        hooks._register_hook('setup', 'Problem', pre=partial(_set_timer_setup_hook, options))

        # register an atexit function to write out all of the timing data
        atexit.register(partial(_postprocess, options))

        with timing_context(not options.use_context):
            _load_and_exec(options.file[0], user_args)

    else:  # assume file is a pickle file
        if options.use_context:
            issue_warning(f"Since given file '{options.file[0]}' is not a python script, the "
                          "'--use_context' option is ignored.")
        _show_view(options.file[0], options)
