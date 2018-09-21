from __future__ import print_function

import sys
import os
import argparse
import time
from collections import defaultdict

from openmdao.devtools.iprof_utils import find_qualified_name, func_group, \
     _collect_methods, _setup_func_group, _get_methods, _Options
from openmdao.utils.mpi import MPI


_registered = False


# def _trace_mem_call(frame, arg, stack, context):
#     """
#     Called whenever a function is called that matches glob patterns and isinstance checks.
#     """
#     memstack, _ = context
#     memstack.append([(frame.f_code.co_filename,
#                      frame.f_code.co_firstlineno,
#                      frame.f_code.co_name), mem_usage()])


# def _trace_mem_ret(frame, arg, stack, context):
#     """
#     Called whenever a function returns that matches glob patterns and isinstance checks.
#     """
#     memstack, mem_changes = context
#     key, mem_start = memstack.pop()
#     usage = mem_usage()
#     delta = usage - mem_start
#     if delta > 0.0:
#         mem_changes[key][0] += delta
#         mem_changes[key][1] += 1
#         if memstack:
#             mem_changes[key][2].add(memstack[-1][0])
#         # print("%g (+%g) MB %s:%d:%s" % (usage, delta,
#         #                                 key[0], key[1], key[2]))


def _create_profile_callback(stream):

    from openmdao.core.system import System
    class_insts = defaultdict(dict)

    def _wrapped(frame, event, arg):
        fname = frame.f_code.co_filename
        if fname[0] != '<' and 'site-packages' not in fname and 'lib/python' not in fname:
            if event == 'call' or event == 'return':
                if 'self' in frame.f_locals:
                    self = frame.f_locals['self']
                    class_ = self.__class__.__name__
                    ident = id(self)
                    d = class_insts[class_]
                    if ident in d:
                        inst_num = d[ident]
                    else:
                        d[ident] = inst_num = len(d)
                    if isinstance(self, System):
                        try:
                            name = self.pathname
                            if not name:
                                name = "''"
                        except AttributeError:
                            name = '?'
                    else:
                        name = '?'
                else:
                    class_ = name = '?'
                    inst_num = -1

                print(event, os.path.abspath(frame.f_code.co_filename), frame.f_code.co_firstlineno,
                      frame.f_code.co_name, mem_usage(), time.time(), class_, name, inst_num,
                      flush=True, file=stream)

    return _wrapped


def _setup(options):
    global _registered, _trace_memory, mem_usage
    if not _registered:
        from openmdao.devtools.memory import mem_usage

        _trace_memory = _create_profile_callback(open(options.fname, "w"))
        _registered = True


def setup(methods=None):
    """
    Setup memory profiling.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or None
        Methods to be profiled, based on glob patterns and isinstance checks.
    """
    _setup(_Options(methods=methods))


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


def _mem_prof_setup_parser(parser):
    parser.add_argument('-f', '--file', action='store', dest='fname',
                        default='mem_trace.raw',
                        help='Dump function trace with memory usage to given file.')
    parser.add_argument('file', metavar='file', nargs=1,
                        help='Python file to profile.')


def _mem_prof_exec(options):
    """
    Process command line args and perform memory profiling on a specified python file.
    """

    progname = options.file[0]
    sys.path.insert(0, os.path.dirname(progname))

    globals_dict = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    if not MPI or MPI.COMM_WORLD.rank == 0:
        _setup(options)
        with open(progname, 'rb') as fp:
            code = compile(fp.read(), progname, 'exec')
        start()
    else:
        with open(progname, 'rb') as fp:
            code = compile(fp.read(), progname, 'exec')

    exec (code, globals_dict)



def postprocess_memtrace(fname):
    from openmdao.utils.general_utils import simple_warning
    from six import iteritems

    info = {}
    by_qual = {}
    cache = {}

    top = None
    stack = []
    maxmem = 0.0

    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if i > 0:   # skip first line, which is return from profile start()
                parts = line.split()
                event, fpath, lineno, func, mem, elapsed, class_, objname, instnum = parts
                lineno = parts[2] = int(lineno)
                mem = parts[4] = float(mem)
                if mem > maxmem:
                    maxmem = mem
                elapsed = parts[5] = float(elapsed)
                instnum = parts[8] = int(instnum)
                key = (fpath, lineno, objname)

                if event == 'call':
                    if func.startswith('<'):
                        qualname = func
                    else:
                        qualname = find_qualified_name(fpath, lineno, cache, full=True)

                    if qualname == '<module>':
                        qualname = fpath + ':' + qualname

                    if objname == "?" and class_ != "?" and instnum >=0:
                        objname = "%s#%d" % (class_, instnum)

                    if key not in info:
                        info[key] = defaultdict(lambda: None)
                        info[key]['calls'] = 0
                        info[key]['fpath'] = fpath
                        info[key]['line'] = lineno
                        info[key]['func'] = func
                        info[key]['total_mem'] = 0.
                        info[key]['time'] = elapsed
                        info[key]['class'] = class_
                        info[key]['objname'] = objname
                        info[key]['qualname'] = qualname
                        info[key]['children'] = []
                        info[key]['child_ids'] = set()

                        by_qual[qualname, objname] = info[key]

                    info[key]['calls'] += 1
                    info[key]['start_mem'] = mem

                    if stack:
                        _, pfpath, plineno, _, _, _, _, pobjname, _ = stack[-1]
                        parent = info[pfpath, plineno, pobjname]
                        ident = id(info[key])
                        if ident not in parent['child_ids']:
                            parent['children'].append(info[key])
                            parent['child_ids'].add(ident)

                    stack.append(parts)
                elif event == 'return':
                    try:
                        (c_event, c_fpath, c_lineno, c_func, c_mem, c_elapsed,
                         c_class, c_obj, c_instnum) = stack.pop()
                    except IndexError:
                        pass  # last line is a return from funct called before we start recording

                    assert c_event == 'call'

                    # there are cases where sometimes a call is recorded but not a return,
                    # so we have to deal with those  :(

                    if c_fpath != fpath or c_lineno != lineno:
                        simple_warning("No return found for %s line %d" % (c_fpath, c_lineno))
                        while c_fpath != fpath or c_lineno != lineno:
                            try:
                                (c_event, c_fpath, c_lineno, c_func, c_mem,
                                c_elapsed, c_class, c_obj, c_instnum) = stack.pop()
                            except IndexError:
                                break

                    if c_fpath == fpath and c_lineno == lineno and c_class == class_:
                        c_key = (c_fpath, c_lineno, c_obj)
                        if c_obj != objname and c_instnum == instnum:
                            obj = info[c_key]
                            del info[c_key]
                            info[key] = obj
                            c_key = key

                        if c_key == key:
                            info[key]['total_mem'] += (mem - c_mem)

                else:
                    raise RuntimeError("event is not 'call' or 'return'")

    srt = sorted(info.values(), key=lambda x: x['total_mem'])
    for val in srt:
        print(val['qualname'], val['objname'], val['total_mem'], val['calls'])

    val = srt[-1]  # by_qual['Problem.final_setup', 'Problem#0']

    min_mem = .1
    seen = set()
    stack = [('', [val])]
    while stack:
        indent, vals = stack.pop()
        for val in vals:
            #if val['total_mem'] > min_mem:
            print("%s%s" % (indent, val['qualname']), val['objname'], val['total_mem'], val['calls'])
            # kids = [c for c in val['children'] if id(c) not in seen]
            # seen.update(id(k) for k in kids)
            if id(val) not in seen:
                seen.add(id(val))
                stack.append((indent + '  ', val['children']))

    print("Max mem usage:", maxmem)

if __name__ == '__main__':
    import sys
    postprocess_memtrace(sys.argv[1])

