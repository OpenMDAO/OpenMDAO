from __future__ import print_function

import os
import sys
import re
import argparse
from contextlib import contextmanager
from collections import defaultdict, OrderedDict

from six import string_types
from six.moves import cStringIO
from numpy import ndarray

from openmdao.devtools.iprof_utils import _create_profile_callback, find_qualified_name, \
                                         func_group, _collect_methods, _Options, _setup_func_group,\
                                         _get_methods

_trace_calls = None  # pointer to function that implements the trace
_registered = False  # prevents multiple atexit registrations

MAXLINE = 80
tab = '    '

addr_regex = re.compile(" at 0x[0-9a-fA-F]+")

def _indented_print(f_locals, d, indent, excludes=set(['__init__', 'self']), file=sys.stdout):
    """
    Print trace info, indenting based on call depth.
    """
    sindent = tab * indent
    sep = '=' if d is f_locals else ':'

    for name in sorted(d, key=lambda a: str(a)):
        if name not in excludes:
            if isinstance(d[name], (dict, OrderedDict)):
                f = cStringIO()
                _indented_print(f_locals, d[name], 0, file=f)
                s = "  %s%s%s{%s}" % (sindent, name, sep, f.getvalue())
            else:
                s = "  %s%s%s%s" % (sindent, name, sep, d[name])
            if ' object at ' in s:
                s = addr_regex.sub('', s)
            linelen = len(s)
            leneq = len(s.split(sep, 1)[0])
            if linelen > MAXLINE:
                if '\n' in s:
                    # change indent
                    s = s.replace("\n", "\n%s" % (' '*leneq))
            print(s, file=file)


def _trace_call(frame, arg, stack, context):
    """
    This is called after we have matched based on glob pattern and isinstance check.
    """
    qual_cache, method_counts, class_counts, verbose = context
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

    indent = tab * (len(stack)-1)
    if verbose:
        print("%s%s (%d)" % (indent, fullname, method_counts[fullname]))
        _indented_print(frame.f_locals, frame.f_locals, len(stack)-1)
    else:
        print("%s%s" % (indent, fullname))
    sys.stdout.flush()


def _trace_return(frame, arg, stack, context):
    """
    This is called when a matched function returns.

    This only happens if show_return is True when setup() is called.
    """
    qual_cache, method_counts, class_counts, _ = context
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

    indent = tab * (len(stack)-1)
    print("%s<-- %s" % (indent, '.'.join((sname, funcname))))
    if arg is not None:
        s = "%s     %s" % (indent, arg)
        if ' object at ' in s:
            s = addr_regex.sub('', s)
        print(s)

    sys.stdout.flush()


def _setup(options):
    if not func_group:
        _setup_func_group()

    global _registered, _trace_calls

    verbose=options.verbose
    if not _registered:
        methods = _get_methods(options, default='openmdao')

        call_stack = []
        qual_cache = {}
        method_counts = defaultdict(int)
        class_counts = defaultdict(set)
        if verbose:
            do_ret = _trace_return
        else:
            do_ret = None
        _trace_calls = _create_profile_callback(call_stack, _collect_methods(methods),
                                                do_call=_trace_call,
                                                do_ret=do_ret,
                                                context=(qual_cache, method_counts,
                                                         class_counts, verbose))


def setup(methods=None, verbose=None):
    """
    Setup call tracing.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or None
        Methods to be traced, based on glob patterns and isinstance checks.
    verbose : bool
        If True, show function locals and return values.
    """
    _setup(_Options(methods=methods, verbose=verbose))


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
def tracing(methods=None, verbose=False):
    """
    Turn on call tracing within a certain context.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or str or None
        Methods to be traced, based on glob patterns and isinstance checks. If value
        is a string, use that string to lookup a 'canned' method list by name.
    verbose : bool
        If True, show function locals and return values.
    """
    setup(methods=methods, verbose=verbose)
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
    verbose : bool
        If True, show function locals and return values.
    """
    def __init__(self, methods=None, verbose=False):
        self.options = _Options(methods=methods, verbose=verbose)
        self._call_setup = True

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            if self._call_setup:
                _setup(self.options)
                self._call_setup = False
            start()
            func(*args, **kwargs)
            stop()
        return wrapped


def _itrace_setup_parser(parser):
    if not func_group:
        _setup_func_group()

    parser.add_argument('file', nargs=1, help='Python file to be traced.')
    parser.add_argument('-g', '--group', action='store', dest='methods',
                        default='openmdao',
                        help='Determines which group of methods will be traced. Default is "openmdao".'
                              ' Options are: %s' % sorted(func_group.keys()))
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help="Show function locals and return values.")


def _itrace_exec(options):
    """
    Process command line args and perform tracing on a specified python file.
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

    _setup(options)
    start()

    exec (code, globals_dict)
