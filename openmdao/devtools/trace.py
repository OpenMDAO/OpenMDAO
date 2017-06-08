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
from openmdao.core.problem import Problem
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver
from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.matrix import Matrix

# This maps a simple identifier to a group of classes and corresponding
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
                    fullname = _code2funcname(self, frame.f_code)
                    _callstack.append(fullname)
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
                    _callstack.pop()

def _trace_memory(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _active_traces[func_name]):
                    _callstack.append((self, frame.f_code, mem_usage()))
    elif event == 'return':
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _active_traces[func_name]):
                    _, code_obj, mem_start = _callstack.pop()
                    delta = mem_usage() - mem_start
                    if delta > 0.0:
                        if code_obj in _mem_changes:
                            _mem_changes[(self, code_obj)] += delta
                        else:
                            _mem_changes[(self, code_obj)] = delta


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
                                                                 code_obj.co_filename))
        classes = [self.__class__.__name__]

    try:
        path = self.pathname
    except AttributeError:
        path = None

    try:
        fname = '.'.join((classes[0], code_obj.co_name))
    except IndexError:
        fname = code_obj.co_name

    if path is None:
        return fname
    else:
        return "%s.<%s>" % (path, fname)

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


def trace_init(trace_type='call', trace_group=True):
    global _registered, trace_mem, trace
    if not _registered:
        if trace_type == 'mem':
            trace_mem = trace_group
            def print_totals():
                items = sorted(_mem_changes.items(), key=lambda x: x[1])
                for (self, code_obj), delta in items:
                    if delta > 0.0:
                        print("%s %g MB" % (_code2funcname(self, code_obj), delta))
            import atexit
            atexit.register(print_totals)
            _registered = True
        else:
            trace = True


def trace_on(class_group='openmdao'):
    global _active_traces, _file2class
    _active_traces, _file2class = _collect_methods(_trace_dict[class_group])
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    sys.setprofile(_trace_calls)


def trace_off():
    sys.setprofile(None)

trace_mem_off = trace_off


def trace_mem_on(class_group='openmdao'):
    global _active_traces, _file2class
    _active_traces, _file2class = _collect_methods(_trace_dict[class_group])
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    sys.setprofile(_trace_memory)


@contextmanager
def tracing(trace_type='call', trace_group='openmdao'):
    trace_init(trace_type, trace_group=trace_group)
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
    trace_group : str, optional
        Identifier of a group of classes and methods to trace.
    """
    def __init__(self, trace_type='call', trace_group='openmdao'):
        self.trace_type = trace_type
        self.trace_group = trace_group

    def __call__(self, func):
        trace_init(trace_type=self.trace_type, trace_group=self.trace_group)

        def wrapped(*args, **kwargs):
            trace_on(self.trace_group)
            func(*args, **kwargs)
            trace_off()
        return wrapped


def main():
    from optparse import OptionParser
    usage = "trace.py [scriptfile [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-t', '--type', dest="type_",
        help="Type of trace (print calls or trace mem usage)", default='call')
    parser.add_option('-g', '--group', dest="group",
        help="Name of class/method group that determines which classes/methods to trace.",
        default='openmdao')

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    if len(args) > 0:
        progname = args[0]
        sys.path.insert(0, os.path.dirname(progname))
        if options.type_ == 'mem':
            global mem_usage
            from openmdao.devtools.debug import mem_usage

        trace_init(trace_type=options.type_, trace_group=options.group)
        with open(progname, 'rb') as fp:
            code = compile(fp.read(), progname, 'exec')
        globs = {
            '__file__': progname,
            '__name__': '__main__',
            '__package__': None,
            '__cached__': None,
        }
        if options.type_ == 'call':
            trace_on(options.group)
        else:
            trace_mem_on(options.group)

        exec (code, globs)

if __name__ == '__main__':
    main()
