
import sys
import atexit
import pickle
from time import perf_counter
from contextlib import contextmanager

import numpy as np
from functools import wraps, partial

import openmdao.utils.hooks as hooks
from openmdao.utils.file_utils import _load_and_exec, _to_filename
from openmdao.utils.mpi import MPI
from openmdao.visualization.timing_viewer.timing_viewer import view_timing, _timing_iter, \
    _timing_file_iter

from openmdao.core.parallel_group import ParallelGroup

# can use this to globally turn timing on/off so we can time specific sections of code
_timing_active = True

class FuncTimer(object):
    """
    Keep track of execution times for a function.
    """
    def __init__(self, name,):
        self.name = name
        self.ncalls = 0
        self.start = 0
        self.min = 1e99
        self.max = 0
        self.tot = 0

    def tick(self):
        if _timing_active:
            self.start = perf_counter()

    def tock(self):
        if _timing_active:
            dt = perf_counter() - self.start
            if dt < self.min:
                self.min = dt
            if dt > self.max:
                self.max = dt
            self.tot += dt
            self.ncalls += 1

    def avg(self):
        if self.ncalls > 0:
            return self.tot / self.ncalls
        return 0.


def _timer_wrap(f, timer):
    """
    Wrap a method to keep track of its execution time.

    Parameters
    ----------
    f : method
        The method being wrapped.
    timer : Timer
        Object to keep track of timing data.
    """
    def do_timing(*args, **kwargs):
        timer.tick()
        ret = f(*args, **kwargs)
        timer.tock()
        return ret

    return wraps(f)(do_timing)


class TimingManager(object):
    def __init__(self):
        self._timers = {}

    def add_timings(self, name_obj_iter, method_names):
        for name, obj in name_obj_iter:
            for method_name in method_names:
                self.add_timing(name, obj, method_name)

    def add_timing(self, name, obj, method_name):
        method = getattr(obj, method_name, None)
        if method is not None:
            if name not in self._timers:
                self._timers[name] = []
            timer = FuncTimer(method_name)
            self._timers[name].append((timer, isinstance(obj, ParallelGroup)))
            setattr(obj, method_name, _timer_wrap(method, timer))


_default_timer_methods = sorted(['_solve_nonlinear', '_apply_nonlinear', '_solve_linear',
                                 '_apply_linear', '_linearize'])


_timing_managers = {}
_timer_methods = None  # TODO: use kwargs instead after Herb's PR goes in


@contextmanager
def timing_context(active):
    """
    Context manager to set whether timing is active or not.

    Parameters
    ----------
    active : bool
        Is timing active or inactive?
    """
    global _timing_active

    save = _timing_active
    _timing_active = bool(active)
    try:
        yield
    finally:
        _timing_active = save


def _setup_sys_timers(system, method_names=tuple(_default_timer_methods)):
    global _timing_managers

    probname = system._problem_meta['name']
    if probname not in _timing_managers:
        _timing_managers[probname] = TimingManager()
    tmanager = _timing_managers[probname]
    name_sys = ((s.pathname, s) for s in system.system_iter(include_self=True, recurse=True))
    tmanager.add_timings(name_sys, method_names)


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
                        help='Name of pickled output file. By default it goes to "timings.pkl".')
    parser.add_argument('-f', '--func', action='append', default=[],
                        dest='funcs', help='Time a specified function. Can be applied multiple '
                        'times to specify multiple functions. '
                        f'Default methods are {_default_timer_methods}.')
    parser.add_argument('-v', '--view', action='store', dest='view', default='browser',
                        help='View the output.  Default view is "browser".  Other options are '
                        '"text" for ascii output or "none" for no output.')


def _setup_timer_hook(system):
    global _timer_methods, _timing_managers

    tmanager = _timing_managers.get(system._problem_meta['name'])
    if tmanager is not None and not tmanager._timers:
        _setup_sys_timers(system, method_names=_timer_methods)


def _set_timer_setup_hook(problem):
    global _timing_managers

    # this just sets a hook into the top level system of the model after we know it exists.
    inst_id = problem._get_inst_id()
    if inst_id not in _timing_managers:
        _timing_managers[inst_id] = TimingManager()
        hooks._register_hook('_setup_procs', 'System', inst_id='', post=_setup_timer_hook)
        hooks._setup_hooks(problem.model)


def _to_ascii(timing_iter, f):
    for rank, probname, sysname, _, parallel, method, ncalls, avg, min, max, tot in timing_iter:
        parallel = '(parallel)' if parallel else ''
        print(f"{rank:4} (rank) {ncalls:7} (calls) {min:12.6f} (min) "
              f"{max:12.6f} (max) {avg:12.6f} (avg) {tot:12.6f} "
              f"(tot) {parallel} {probname} {sysname}:{method}", file=f)


def _postprocess(timing_file, options):
    global _timer_methods, _timing_managers

    if timing_file is None:
        timing_file = 'timings.pkl'

    if MPI is not None:
        # need to consolidate the timing data from different procs
        all_managers = MPI.COMM_WORLD.gather(_timing_managers, root=0)
        if MPI.COMM_WORLD.rank == 0:
            pass
        else:
            return
    else:
        all_managers = [_timing_managers]

    with open(timing_file, 'wb') as f:
        print(f"Saving timing data to '{timing_file}'.")
        pickle.dump(all_managers, f, pickle.HIGHEST_PROTOCOL)

    view = options.view.lower()

    if view == 'browser':
        view_timing(timing_file, outfile='timing_report.html', show_browser=True)
    elif view == 'text':
        _to_ascii(_timing_iter(all_managers), sys.stdout)
    elif view == 'none':
        pass
    else:
        print(f"\nViewing option '{view}' ignored.  Doing nothing.  Valid options are "
              "['browser', 'text', 'none'].")


def _timing_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao timing'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    global _timer_methods, _timing_managers

    filename = _to_filename(options.file[0])
    if filename.endswith('.py'):
        _timer_methods = options.funcs
        if not _timer_methods:
            _timer_methods = _default_timer_methods.copy()

        hooks._register_hook('setup', 'Problem', pre=_set_timer_setup_hook)

        if options.outfile is not None and MPI:
            outfile = f"{options.outfile}.{MPI.COMM_WORLD.rank}"
        else:
            outfile = options.outfile

        # register an atexit function to write out all of the timing data
        atexit.register(partial(_postprocess, outfile, options))

        _load_and_exec(options.file[0], user_args)
    else:  # assume file is a pickle file
        view = options.view.lower()

        if view == 'browser':
            view_timing(options.file[0], outfile='timing_report.html', show_browser=True)
        elif view == 'text':
            _to_ascii(_timing_file_iter(options.file[0]), sys.stdout)
        elif view == 'none':
            pass
        else:
            print(f"\nViewing option '{view}' ignored.  Doing nothing.  Valid options are "
                "['browser', 'text', 'none'].")
