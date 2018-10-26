from __future__ import print_function

import sys
import os
import argparse
import time
from fnmatch import fnmatch
from collections import defaultdict
from os.path import abspath, isfile

from openmdao.devtools.iprof_utils import find_qualified_name, func_group, \
     _collect_methods, _setup_func_group, _get_methods, _Options
from openmdao.utils.mpi import MPI
from openmdao.devtools.debug import _get_color_printer


_registered = False


def _create_profile_callback(stream, globs=None):

    from openmdao.core.system import System
    class_insts = defaultdict(dict)

    def _wrapped(frame, event, arg):
        fname = frame.f_code.co_filename
        # exclude any c calls and any builtin python '<func>' functions
        if fname[0] != '<' and (event == 'call' or event == 'return'):
            if globs:
                for g in globs:
                    if fnmatch(fname, g):
                        break
                else:
                    return
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

            print(event, abspath(fname), frame.f_code.co_firstlineno,
                    frame.f_code.co_name, mem_usage(), time.time(), class_, name, inst_num,
                    sep='|', file=stream)

    return _wrapped


def _setup(options):
    global _registered, _trace_memory, mem_usage, _out_stream
    if not _registered:
        from openmdao.devtools.memory import mem_usage

        _out_stream = open(options.outfile, "w", buffering=1024*1024)
        _trace_memory = _create_profile_callback(_out_stream, options.globs)
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
    global _out_stream
    sys.setprofile(None)
    _out_stream.close()


def _mem_prof_setup_parser(parser):
    parser.add_argument('-o', '--outfile', action='store', dest='outfile', default='mem_trace.raw',
                        help='Name of file containing memory dump.')
    parser.add_argument('--min', action='store', dest='min_mem', type=float, default=1.0,
                        help='Dump function trace with memory usage in MB above min_mem to the '
                        'given file. Default is 1.0.')
    parser.add_argument('-c', '--colors', action='store_true', dest='show_colors',
                        help="Display colors if the terminal supports it.  Requires 'colorama' "
                             "python package.  Use 'pip install colorama' to install it.")
    parser.add_argument('-g', '--glob', action='append', dest='globs', default=[],
                        help='Glob patterns to match source files to be traced.')
    parser.add_argument('file', metavar='file', nargs=1, help='Python file to profile.')


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

    stop()

    postprocess_memtrace(fname=options.outfile, min_mem=options.min_mem,
                         show_colors=options.show_colors)


def _mempost_setup_parser(parser):
    parser.add_argument('--out', action='store', dest='outfile', default=None,
                        help='Dump memory tree to given file.')
    parser.add_argument('--min', action='store', dest='min_mem', type=float, default=1.0,
                        help='Dump function trace with memory usage to given file.')
    parser.add_argument('-c', '--colors', action='store_true', dest='show_colors',
                        help="Display colors if the terminal supports it.  Requires 'colorama' "
                             "python package.  Use 'pip install colorama' to install it.")
    parser.add_argument('file', metavar='file', nargs=1, help='Memory dump file to process.')


def _mempost_exec(options):
    """
    Process command line args and perform postprocessing on the specified memory dump file.
    """
    if options.outfile in ('sys.stdout', None):
        stream = sys.stdout
    else:
        stream = open(options.outfile, 'w')

    postprocess_memtrace(fname=options.file[0], min_mem=options.min_mem,
                         show_colors=options.show_colors, stream=stream)


def _process_parts(parts):
    event, fpath, lineno, func, mem, elapsed, class_, objname, instnum = parts[:9]
    lineno = parts[2] = int(lineno)
    mem = parts[4] = float(mem)
    elapsed = parts[5] = float(elapsed)
    instnum = parts[8] = int(instnum)

    if instnum >= 0:
        class_ = "%s#%d" % (class_, instnum)
        if objname == '?':
            objname = class_
        else:
            objname = "%s:%s" % (class_, objname)
    else:
        objname = class_ = ''

    parts[7] = objname

    call = "%s:%d:%s%s" % (fpath, lineno, class_, func)
    parts.append(call)

    return parts


def postprocess_memtrace(fname, min_mem=1.0, show_colors=True, rank=0, stream=sys.stdout):
    from openmdao.utils.general_utils import simple_warning
    cprint, Fore, Back, Style = _get_color_printer(stream, show_colors, rank=rank)

    info = {}
    cache = {}

    top = None
    stack = []
    path_stack = []
    maxmem = 0.0

    if stream is None:
        stream = sys.stdout

    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if i > 0:   # skip first line, which is return from profile start()
                parts = _process_parts(line.split('|'))
                event, fpath, lineno, func, mem, elapsed, class_, objname, instnum, call = parts

                if mem > maxmem:
                    maxmem = mem

                if event == 'call':

                    path_stack.append(call)
                    call_path = '|'.join(path_stack)

                    key = call_path

                    if key not in info:
                        if not func[0] == '<' and isfile(fpath):
                            qualname = find_qualified_name(fpath, lineno, cache, full=True)
                        else:
                            qualname = "%s:%d:%s" % (fpath, lineno, func)

                        info[key] = idict = {
                            'calls': 0,
                            'fpath': fpath,
                            'line': lineno,
                            'func': func,
                            'total_mem': 0.,
                            'time': elapsed,
                            'class': class_,
                            'objname': objname,
                            'qualname': qualname,
                            'children': [],
                            'child_ids': set(),
                            'call_path': call_path,
                        }
                    else:
                        idict = info[key]

                    idict['calls'] += 1

                    if stack:
                        pcall_path = '|'.join(path_stack[:-1])
                        parent = info[pcall_path]
                        ident = id(info[key])
                        if ident not in parent['child_ids']:
                            parent['children'].append(info[key])
                            parent['child_ids'].add(ident)

                    stack.append(parts)

                elif event == 'return':
                    try:
                        c_parts = stack.pop()
                    except IndexError:
                        path_stack.pop()
                        continue  # last line is a return from funct called before we start recording

                    c_parts = _process_parts(c_parts[:9])
                    (c_event, c_fpath, c_lineno, c_func, c_mem, c_elapsed, c_class, c_obj,
                     c_instnum, c_call) = c_parts

                    # there are cases where sometimes a call is recorded but not a return,
                    # so we have to deal with those  :(
                    while call != c_call:
                        simple_warning("No return found for %s." % c_call)
                        try:
                            c_parts = stack.pop()
                            path_stack.pop()
                        except IndexError:
                            continue

                        c_parts = _process_parts(c_parts)
                        if len(c_parts) != 10:
                            print("bad_line:", c_parts)
                            continue
                        (c_event, c_fpath, c_lineno, c_func, c_mem, c_elapsed, c_class, c_obj,
                         c_instnum, c_call) = c_parts

                    info['|'.join(path_stack)]['total_mem'] += (mem - c_mem)

                    path_stack.pop()

    srt = sorted(info.values(), key=lambda x: x['total_mem'])

    val = srt[-1]

    seen = set()
    stack = [('', iter([val]))]
    while stack:
        indent, children = stack[-1]
        try:
            val = next(children)
            if val['total_mem'] > min_mem:
                cprint("%s%7.2f " % (indent, val['total_mem']), color=Fore.GREEN + Style.BRIGHT)
                cprint(" (%d calls)  " % val['calls'])
                cprint("%s %s\n" % (val['qualname'], val['objname']))
                if id(val) not in seen:
                    seen.add(id(val))
                    stack.append((indent + '  ', iter(val['children'])))
        except StopIteration:
            stack.pop()

    cprint("\nMax mem usage: ")
    cprint("%7.2f MB\n" % maxmem, color=Fore.RED + Style.BRIGHT)

if __name__ == '__main__':
    import sys
    postprocess_memtrace(sys.argv[1])

