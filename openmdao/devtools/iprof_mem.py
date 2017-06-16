from __future__ import print_function

import sys
import os
import atexit

from openmdao.devtools.prof_utils import _create_profile_callback, find_qualified_name, func_group, \
     _collect_methods


_registered = False  # prevents multiple atexit registrations


def _trace_mem_call(frame, arg, stack, context):
    memstack, _ = context
    memstack.append([frame.f_code, mem_usage()])


def _trace_mem_ret(frame, arg, stack, context):
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
    global _registered, _trace_memory, mem_usage
    if not _registered:
        from openmdao.devtools.debug import mem_usage
        if methods is None:
            methods = func_group['openmdao']

        mem_changes = {}
        memstack = []
        callstack = []
        _trace_memory = _create_profile_callback(callstack,  _collect_methods(methods),
                                                 _trace_mem_call, _trace_mem_ret,
                                                 (memstack, mem_changes))

        _qual_cache = {}  # cache of files scanned for qualified names

        def print_totals():
            for (self, code_obj), delta in sorted(mem_changes.items(), key=lambda x: x[1]):
                if delta != 0.0:
                    qfile, qclass, qname = find_qualified_name(code_obj.co_filename,
                                                               code_obj.co_firstlineno,
                                                               _qual_cache)
                    if qfile is None:
                        fullname = '.'.join((qclass, qname))
                    else:
                        fullname = ':'.join((qfile, qname))
                    print("%s %g MB" % (fullname, delta))

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


def main():
    from optparse import OptionParser
    usage = "iprof_mem.py [scriptfile [arg] ..."
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
