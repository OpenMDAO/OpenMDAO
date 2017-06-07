from __future__ import print_function

import os
import sys
import warnings

from inspect import getmembers, isroutine
from fnmatch import fnmatchcase
from contextlib import contextmanager
from collections import defaultdict

from six import iteritems, PY3

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

# This maps a simple identifier to a group of classes and possibly corresponding
# glob patterns for each class.
_trace_dict = {
    'openmdao': {
        "*": (System, Jacobian, Matrix, Solver, Driver, Problem),
    },
    'setup': {
        "*setup*": (System, Solver, Driver, Problem),
    },
}


_active_traces = {}
_file2class = {}
_method_counts = {}
_mem_changes = {}
_callstack = []
_registered = False  # prevents multiple atexit registrations

# even through this is called '_trace_calls', it's called as part of
# sys.setprofile and not sys.settrace because we're not interested in
# tracing by line.
def _trace_calls(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _active_traces[func_name]):
                    classes = _file2class[frame.f_code.co_filename]
                    if not classes:
                        for base in self.__class__.__mro__:
                            if base is object:
                                continue
                            if PY3:
                                fname = sys.modules[base.__module__].__file__
                            else:
                                fname = sys.modules[base.__module__].__file__[:-1]
                            classes = _file2class[fname]
                            if base.__name__ not in classes:
                                classes.append(base.__name__)
                    if len(classes) > 1:
                        # TODO: fix this
                        warnings.warn("multiple classes %s found in same module (%s), "
                                      "using self.__class__ (might be wrong)" % (classes,
                                                                                 frame.f_code.co_filename))
                        classes = [self.__class__.__name__]
                    try:
                        fullname = '.'.join((classes[0], func_name))
                    except IndexError:
                        fullname = func_name
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
    elif event == 'return':
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _active_traces[func_name]):
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

def _trace_memory(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _active_traces[func_name]):
                    classes = _file2class[frame.f_code.co_filename]
                    if not classes:
                        for base in self.__class__.__mro__:
                            if base is object:
                                continue
                            if PY3:
                                fname = sys.modules[base.__module__].__file__
                            else:
                                fname = sys.modules[base.__module__].__file__[:-1]
                            classes = _file2class[fname]
                            if base.__name__ not in classes:
                                classes.append(base.__name__)
                    if len(classes) > 1:
                        # TODO: fix this
                        warnings.warn("multiple classes %s found in same module (%s), "
                                      "using self.__class__ (might be wrong)" % (classes,
                                                                                 frame.f_code.co_filename))
                        classes = [self.__class__.__name__]
                    try:
                        fullname = '.'.join((classes[0], func_name))
                    except IndexError:
                        fullname = func_name

                    _callstack.append((fullname, mem_usage()))
    elif event == 'return':
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _active_traces[func_name]):
                    fullname, mem_start = _callstack.pop()
                    delta = mem_usage() - mem_start
                    if delta > 0.0:
                        if fullname in _mem_changes:
                            _mem_changes[fullname] += delta
                        else:
                            _mem_changes[fullname] = delta


def _code2funcname(self, code_obj):
    classes = _file2class[code_obj.co_filename]
    if not classes:
        for base in self.__class__.__mro__:
            if base is object:
                continue
            if PY3:
                fname = sys.modules[base.__module__].__file__
            else:
                fname = sys.modules[base.__module__].__file__[:-1]
            classes = _file2class[fname]
            if base.__name__ not in classes:
                classes.append(base.__name__)
    if len(classes) > 1:
        # TODO: fix this
        warnings.warn("multiple classes %s found in same module (%s), "
                      "using self.__class__ (might be wrong)" % (classes,
                                                                 frame.f_code.co_filename))
        classes = [self.__class__.__name__]
    try:
        return '.'.join((classes[0], func_name))
    except IndexError:
        return func_name


def _collect_methods(method_dict):
    """
    Iterate over a dict of method name patterns mapped to classes.  Search
    through the classes for anything that matches and return a dict of
    exact name matches and their correspoding classes.

    Parameters
    ----------
    method_dict : {pattern1: classes1, ... pattern_n: classes_n}
        Dict of glob patterns mapped to lists of classes used for isinstance checks

    Returns
    -------
    dict
        Dict of method names and tuples of all classes that matched for that method.
    """
    matches = {}
    file2class = defaultdict(list)  # map files to classes

    # TODO: update this to also work with stand-alone functions
    for pattern, classes in iteritems(method_dict):
        for class_ in classes:
            for base in class_.__mro__:
                if base is object:
                    continue
                if PY3:
                    fname = sys.modules[base.__module__].__file__
                else:
                    fname = sys.modules[base.__module__].__file__[:-1]
                classes = file2class[fname]
                if base.__name__ not in classes:
                    file2class[fname].append(base.__name__)

            for name, obj in getmembers(class_):
                if callable(obj) and (pattern == '*' or fnmatchcase(name, pattern)):
                    if name in matches:
                        matches[name].append(class_)
                    else:
                        matches[name] = [class_]

    # convert values to tuples so we can use in isinstance call
    for name in matches:
        matches[name] = tuple(matches[name])

    return matches, file2class


def trace_init(trace_type='call', trace_mem_group=True):
    global _registered, trace_mem, trace
    if not _registered:
        if trace_type == 'mem':
            trace_mem = trace_mem_group
            def print_totals():
                items = sorted(_mem_changes.items(), key=lambda x: x[1])
                for n, delta in items:
                    if delta > 0.0:
                        print("%s %g MB" % (n, delta))
            import atexit
            atexit.register(print_totals)
            _registered = True
        else:
            trace = True


def trace_on(class_group='openmdao'):
    global _active_traces, _file2class
    print("class_group",class_group)
    _active_traces, _file2class = _collect_methods(_trace_dict[class_group])
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    sys.setprofile(_trace_calls)


def trace_off():
    sys.setprofile(None)


@contextmanager
def tracing(trace_type='call', trace_mem_group='openmdao'):
    trace_init(trace_type, trace_mem_group=trace_mem_group)
    trace_on(class_group)
    yield
    trace_off()


class tracedfunc(object):
    """
    Decorator that activates tracing for a particular function.

    Parameters
    ----------
    trace_type : str, optional
        Type of tracing to perform.  Options are ['call', 'mem']
    class_group : str, optional
        Identifier of a group of classes that will have their functions traced.
    """
    def __init__(self, trace_type='call', trace_mem_group='openmdao'):
        self.trace_type = trace_type
        self.trace_mem_group = trace_mem_group

    def __call__(self, func):
        trace_init(trace_type=self.trace_type, trace_mem_group=self.trace_mem_group)

        def wrapped(*args, **kwargs):
            trace_on(self.classes)
            func(*args, **kwargs)
            trace_off()
        return wrapped

trace_calls = os.environ.get('OPENMDAO_TRACE')
trace_mem = os.environ.get('OPENMDAO_TRACE_MEM')

if trace_calls:
    trace_init(trace_type='call')
    trace_on(trace_calls)
elif trace_mem:
    from openmdao.devtools.debug import mem_usage
    trace_init(trace_type='mem', trace_mem_group=trace_mem)
    trace_on(trace_mem)
