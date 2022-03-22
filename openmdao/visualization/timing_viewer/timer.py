import builtins
import pickle

from time import perf_counter
from contextlib import contextmanager

from functools import wraps, partial

import openmdao.utils.hooks as hooks
from openmdao.utils.om_warnings import issue_warning

from openmdao.core.parallel_group import ParallelGroup

# can use this to globally turn timing on/off so we can time specific sections of code
_timing_active = False
_total_time = 0.
_timing_managers = {}

class _RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        # Only allow classes from this module.
        if module == 'openmdao.visualization.timing_viewer.timer':
            return globals().get(name)
        # Forbid everything else.
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden in timing file.")


def _restricted_load(f):
    """Like pickle.load() but restricted to a specific set of classes."""
    return _RestrictedUnpickler(f).load()


def _timing_iter(all_timing_managers):
    for rank, (timing_managers, tot_time) in enumerate(all_timing_managers):
        for probname, tmanager in timing_managers.items():
            for sysname, timers in tmanager._timers.items():
                for t, parallel in timers:
                    if t.ncalls > 0:
                        level = len(sysname.split('.')) if sysname else 0
                        yield rank, probname, sysname, level, parallel, t.name, t.ncalls, t.avg(),\
                            t.min, t.max, t.tot, tot_time


def _timing_file_iter(timing_file):
    with open(timing_file, 'rb') as f:
        yield from _timing_iter(_restricted_load(f))

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
        global _timing_active
        if _timing_active:
            self.start = perf_counter()

    def tock(self):
        global _timing_active
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


@contextmanager
def timing_context(active):
    """
    Context manager to set whether timing is active or not.

    Parameters
    ----------
    active : bool
        Is timing active or inactive?
    """
    global _timing_active, _total_time

    active = bool(active)
    ignore = _timing_active and active
    if ignore:
        issue_warning("Timing is already active outside of this timing_context, so it will be "
                      "ignored.")

    start_time = perf_counter()

    save = _timing_active
    _timing_active = active
    try:
        yield
    finally:
        _timing_active = save
        if active and not ignore:
            _total_time += perf_counter() - start_time


def _setup_sys_timers(system, method_names):
    # decorate all specified System methods
    global _timing_managers

    probname = system._problem_meta['name']
    if probname not in _timing_managers:
        _timing_managers[probname] = TimingManager()
    tmanager = _timing_managers[probname]
    name_sys = ((s.pathname, s) for s in system.system_iter(include_self=True, recurse=True))
    tmanager.add_timings(name_sys, method_names)


def _setup_timers(options, system):
    # hook called after _setup_procs to decorate all specified System methods
    global _timing_managers

    timer_methods = options.funcs

    tmanager = _timing_managers.get(system._problem_meta['name'])
    if tmanager is not None and not tmanager._timers:
        _setup_sys_timers(system, method_names=timer_methods)


def _set_timer_setup_hook(options, problem):
    # this just sets a hook into the top level system of the model after we know it exists.
    global _timing_managers

    inst_id = problem._get_inst_id()
    if inst_id not in _timing_managers:
        _timing_managers[inst_id] = TimingManager()
        hooks._register_hook('_setup_procs', 'System', inst_id='',
                             post=partial(_setup_timers, options))
        hooks._setup_hooks(problem.model)
