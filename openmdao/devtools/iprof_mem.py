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
    memstack, _ = context
    memstack.append([frame.f_code, mem_usage()])


def _trace_mem_ret(frame, arg, stack, context):
    memstack, mem_changes = context
    code_obj, mem_start = memstack.pop()
    usage = mem_usage()
    delta = usage - mem_start
    if delta > 0.0:
        # FIXME: this can allocate memory to store the code_obj data!
        mem_changes[code_obj] += delta

        # print("%g (+%g) MB %s:%d:%s" % (usage, delta,
        #                                 code_obj.co_filename,
        #                                 code_obj.co_firstlineno,
        #                                 code_obj.co_name))

        # we only want to see deltas from the routines that actually allocate
        # memory rather than those routines and all of the routines that call
        # them either directly or indirectly, so we add the current delta to
        # the mem usage up the call stack, which will subtract it from the ancestor
        # deltas.
        for i in range(len(memstack)):
            memstack[i][1] += delta


def setup(methods=None):
    global _registered, _trace_memory, mem_usage
    if not _registered:
        from openmdao.devtools.debug import mem_usage
        if methods is None:
            methods = func_group['openmdao_all']

        mem_changes = defaultdict(float)
        memstack = []
        callstack = []
        _trace_memory = _create_profile_callback(callstack,  _collect_methods(methods),
                                                 do_call=_trace_mem_call, do_ret=_trace_mem_ret,
                                                 context=(memstack, mem_changes))

        def print_totals():
            for code_obj, delta in sorted(mem_changes.items(), key=lambda x: x[1]):
                if delta != 0.0:
                    print("%s:%d:%s %g MB" % (code_obj.co_filename,
                                              code_obj.co_firstlineno,
                                              code_obj.co_name,
                                              delta))

        atexit.register(print_totals)
        _registered = True


def start():
    global _trace_memory
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    if _trace_memory is None:
        raise RuntimeError("trace.setup() was not called before trace.start().")
    sys.setprofile(_trace_memory)


def stop():
    sys.setprofile(None)


def profile_py_file():

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
    profile_py_file()
