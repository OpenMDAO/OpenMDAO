from __future__ import print_function

import os
import sys
import gc

from inspect import getmembers, isroutine
from fnmatch import fnmatchcase
from contextlib import contextmanager

from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.vectors.vector import Vector, Transfer
from openmdao.core.problem import Problem
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver
from openmdao.core.component import Component
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.matrix import Matrix

_trace_dict = {
    'openmdao': (System, Component, Group, Problem),
    'setup': [(s, ['*setup*']) for s in (System, Component, Group, Problem)]
}


def _get_methods(klass, patterns=None):
    methods = getmembers(klass, isroutine)
    if patterns is None:
        return [name for name, method in methods]
    else:
        filtered = []
        for name, _ in methods:
            for p in patterns:
                if fnmatchcase(name, p):
                    filtered.append(name)
                    break
        return filtered

def _get_all_methods(class_patterns):
    """
    Return a dict of method names and corresponding class tuples.

    Parameters
    ----------
    class_patterns : list of (class, pattern) tuples
        A list of classes and patters to filter their methods

    Returns
    -------
    dict
        Methods dict.  Values are class tuples.
    """
    methods = {}
    for tup in class_patterns:
        if isinstance(tup, tuple):
            klass, patterns = tup
        else:
            klass = tup
            patterns = None
        for name in _get_methods(klass, patterns):
            if name in methods:
                methods[name].append(klass)
            else:
                methods[name] = [klass]

    # convert class lists to tuples so we can use in isinstance calls
    for name in methods:
        methods[name] = tuple(methods[name])
        # TODO: keep most base of classes in tuple (see if it affects performance)

    return methods


_active_traces = {}
_method_counts = {}
_mem_changes = {}
_callstack = []

def _trace_calls(frame, event, arg):
    func_name = frame.f_code.co_name
    if func_name in _active_traces:
        loc = frame.f_locals
        if 'self' in loc:
            insts = _active_traces[func_name]
            self = loc['self']
            if isinstance(self, insts):
                if event is 'call':
                    fullname = '.'.join((self.__class__.__name__, func_name))

                    if trace_mem:
                        _callstack.append((fullname, mem_usage()))
                    else:
                        _callstack.append(func_name)
                        if fullname in _method_counts:
                            _method_counts[fullname] += 1
                        else:
                            _method_counts[fullname] = 1
                        print('   ' * len(_callstack),
                              "%s (%d)" % (fullname, _method_counts[fullname]))
                elif event is 'return':
                    if trace_mem:
                        fullname, mem_start = _callstack.pop()
                        delta = mem_usage() - mem_start
                        if delta > 0.0:
                            if fullname in _mem_changes:
                                _mem_changes[fullname] += delta
                            else:
                                _mem_changes[fullname] = delta
                    else:
                        _callstack.pop()

def trace_init(trace_type='call'):
    if trace_type == 'mem':
        def print_totals():
            items = sorted(_mem_changes.items(), key=lambda x: x[1])
            for n, delta in items:
                if delta > 0.0:
                    print("%s %g" % (n, delta))
        import atexit
        atexit.register(print_totals)

def trace_on(trace_classes=((System, None),)):
    global _active_traces
    _active_traces = _get_all_methods(trace_classes)
    sys.setprofile(_trace_calls)

def trace_off():
    sys.setprofile(None)

@contextmanager
def tracing(trace_type='call', classes='openmdao'):
    trace_init(trace_type)
    trace_on(classes)
    yield
    trace_off()

# decorator
def tracefunc(fnc):
    def wrapped(*args, **kwargs):
        trace_init(trace_type='call')
        trace_on((object,))
        fnc(*args, **kwargs)
        trace_off()
    return wrapped

trace = os.environ.get('OPENMDAO_TRACE')
trace_mem = os.environ.get('OPENMDAO_TRACE_MEM')

if trace:
    trace_init(trace_type='call')
    trace_on(_trace_dict[trace])
elif trace_mem:
    from openmdao.devtools.debug import mem_usage
    trace_init(trace_type='mem')
    trace_on(_trace_dict[trace_mem])
