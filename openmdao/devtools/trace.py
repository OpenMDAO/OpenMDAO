from __future__ import print_function

import os
import sys

from contextlib import contextmanager
from collections import defaultdict

from openmdao.devtools.prof_utils import _create_profile_callback, find_qualified_name, func_group, _collect_methods


# final dict of method names and their corresponding classes
_trace_calls = None  # pointer to function that implements the trace
_registered = False  # prevents multiple atexit registrations


def _trace_call(frame, arg, stack, context):
    qual_cache, method_counts = context
    qfile, qclass, qname = find_qualified_name(frame.f_code.co_filename,
                                               frame.f_code.co_firstlineno,
                                               qual_cache)
    if qfile is None:
        fullname = '.'.join((qclass, qname))
    else:
        fullname = ':'.join((qfile, qname))

    method_counts[fullname] += 1

    print('   ' * len(stack),
          "%s (%d)" % (fullname, method_counts[fullname]))



def setup(methods=None):
    global _registered, _trace_calls
    if not _registered:
        if methods is None:
            methods = func_group['openmdao']

        call_stack = []
        qual_cache = {}
        method_counts = defaultdict(int)
        _trace_calls = _create_profile_callback(call_stack, _collect_methods(methods), _trace_call,
                                                context=(qual_cache, method_counts))


def start():
    global _trace_calls
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    if _trace_calls is None:
        raise RuntimeError("trace.setup() was not called before trace.start().")
    sys.setprofile(_trace_calls)


def stop():
    sys.setprofile(None)


@contextmanager
def tracing(methods=None):
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
        A group of classes and methods to trace.
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


def main():
    from optparse import OptionParser
    usage = "trace.py [scriptfile [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
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

        setup(methods=func_group[options.group])
        with open(progname, 'rb') as fp:
            code = compile(fp.read(), progname, 'exec')
        globs = {
            '__file__': progname,
            '__name__': '__main__',
            '__package__': None,
            '__cached__': None,
        }

        start()

        exec (code, globs)

if __name__ == '__main__':
    main()
