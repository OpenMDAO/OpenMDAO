from __future__ import print_function

import sys
import os
import atexit
import argparse
from collections import defaultdict

from openmdao.devtools.iprof_utils import _create_profile_callback, find_qualified_name, func_group, \
     _collect_methods


_registered = False  # prevents multiple atexit registrations


def _trace_mem_call(frame, arg, stack, context):
    """
    Called whenever a function is called that matches glob patterns and isinstance checks.
    """
    memstack, _ = context
    memstack.append([frame.f_code, mem_usage()])


def _trace_mem_ret(frame, arg, stack, context):
    """
    Called whenever a function returns that matches glob patterns and isinstance checks.
    """
    memstack, mem_changes = context
    code_obj, mem_start = memstack.pop()
    delta = mem_usage() - mem_start
    if delta > 0.0:
        if code_obj in mem_changes:
            mem_changes[(frame.f_locals['self'], code_obj)] += delta
        else:
            mem_changes[(frame.f_locals['self'], code_obj)] = delta

        # we only want to see deltas from the routines that actually allocate
        # memory rather than those routines and all of the routines that call
        # them either directly or indirectly, so we add the current delta to
        # the mem usage up the call stack, which will subtract it from the ancestor
        # deltas.
        for i in range(len(memstack)):
            memstack[i][1] += delta


def setup(methods=None):
    """
    Setup memory profiling.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or None
        Methods to be profiled, based on glob patterns and isinstance checks.
    """
    global _registered, _trace_memory, mem_usage
    if not _registered:
        from openmdao.devtools.debug import mem_usage
        if methods is None:
            methods = func_group['openmdao_all']

        mem_changes = {}
        memstack = []
        callstack = []
        _trace_memory = _create_profile_callback(callstack,  _collect_methods(methods),
                                                 do_call=_trace_mem_call, do_ret=_trace_mem_ret,
                                                 context=(memstack, mem_changes))

        _qual_cache = {}  # cache of files scanned for qualified names

        def print_totals():
            count = defaultdict(set)
            for (self, code_obj), delta in sorted(mem_changes.items(), key=lambda x: x[1]):
                if delta != 0.0:
                    funcname = find_qualified_name(code_obj.co_filename,
                                                   code_obj.co_firstlineno, _qual_cache)
                    try:
                        pname = '(%s)' % self.pathname
                    except AttributeError:
                        pname = ''

                    cname = self.__class__.__name__
                    count[cname].add(id(self))
                    sname = "%s#%d%s" % (self.__class__.__name__, len(count[cname]), pname)

                    print("%g MB %s" % (delta, '.'.join((sname, funcname))))

        atexit.register(print_totals)
        _registered = True


def start():
    """
    Turn on memory profiling.
    """
    global _trace_memory
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    if _trace_memory is None:
        raise RuntimeError("trace.setup() was not called before trace.start().")
    sys.setprofile(_trace_memory)


def stop():
    """
    Turn off memory profiling.
    """
    sys.setprofile(None)


def _profile_py_file():
    """
    Process command line args and perform memory profiling on a specified python file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--group', action='store', dest='group',
                        default='openmdao_all',
                        help='Determines which group of methods will be tracked. Options are %s' %
                             sorted(func_group.keys()))
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
    _profile_py_file()
