from __future__ import print_function

import os
import sys
import argparse
from contextlib import contextmanager
from collections import defaultdict

from six import string_types

from openmdao.devtools.iprof_utils import _create_profile_callback, find_qualified_name, \
                                         func_group, _collect_methods


_trace_calls = None  # pointer to function that implements the trace
_registered = False  # prevents multiple atexit registrations


def _trace_call(frame, arg, stack, context):
    """
    This is called after we have matched based on glob pattern and isinstance check.
    """
    qual_cache, method_counts, class_counts = context
    funcname = find_qualified_name(frame.f_code.co_filename,
                                   frame.f_code.co_firstlineno, qual_cache)

    self = frame.f_locals['self']
    try:
        pname = "(%s)" % self.pathname
    except AttributeError:
        pname = ""

    cname = self.__class__.__name__
    class_counts[cname].add(id(self))
    sname = "%s#%d%s" % (self.__class__.__name__, len(class_counts[cname]), pname)

    fullname = '.'.join((sname, funcname))
    method_counts[fullname] += 1

    indent = '    ' * (len(stack)-1)
    print("%s%s (%d)" % (indent, fullname, method_counts[fullname]))
    for name in frame.f_locals:
        if frame.f_code.co_name != '__init__':
            s = "%s  %s=%s" % (indent, name, frame.f_locals[name])
            if ' object at ' not in s:
                print(s)

def setup(methods=None):
    """
    Setup call tracing.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or None
        Methods to be traced, based on glob patterns and isinstance checks.

    """
    global _registered, _trace_calls
    if not _registered:
        if methods is None or methods not in func_group:
            methods = func_group['openmdao']
        elif isinstance(methods, string_types):
            methods = func_group[methods]

        call_stack = []
        qual_cache = {}
        method_counts = defaultdict(int)
        class_counts = defaultdict(set)
        _trace_calls = _create_profile_callback(call_stack, _collect_methods(methods),
                                                do_call=_trace_call,
                                                context=(qual_cache, method_counts, class_counts))


def start():
    """
    Start call tracing.
    """
    global _trace_calls
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    if _trace_calls is None:
        raise RuntimeError("trace.setup() was not called before trace.start().")
    sys.setprofile(_trace_calls)


def stop():
    """
    Stop call tracing.
    """
    sys.setprofile(None)


@contextmanager
def tracing(methods=None):
    """
    Turn on call tracing within a certain context.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or None
        Methods to be traced, based on glob patterns and isinstance checks.

    """
    setup(methods=methods)
    start()
    yield
    stop()


class tracedfunc(object):
    """
    Decorator that activates tracing for a particular function.

    Parameters
    ----------
    methods : list of (glob, (classes...)) tuples, optional
        Methods to be traced, based on glob patterns and isinstance checks.
    """
    def __init__(self, methods=None):
        self.methods = methods

    def __call__(self, func):
        setup(methods=self.methods)

        def wrapped(*args, **kwargs):
            start()
            func(*args, **kwargs)
            stop()
        return wrapped


def trace_py_file():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--group', action='store', dest='group',
                        default='openmdao',
                        help='Determines which group of methods will be tracked. Default is "openmdao".'
                              ' Options are: %s' % sorted(func_group.keys()))
    parser.add_argument('file', metavar='file', nargs=1,
                        help='Python file to profile.')

    options = parser.parse_args()

    progname = options.file[0]
    sys.path.insert(0, os.path.dirname(progname))

    setup(methods=func_group[options.group])
    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')

    globals_dict = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    start()

    exec (code, globals_dict)


if __name__ == '__main__':
    trace_py_file()
