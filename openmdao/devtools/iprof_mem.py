
import sys
import os
import argparse
import time
import gc
from importlib import import_module
from collections import defaultdict
from os.path import abspath, isfile, dirname, join
from contextlib import contextmanager

from openmdao.devtools.iprof_utils import find_qualified_name, func_group, \
     _collect_methods, _setup_func_group, _get_methods, _Options
from openmdao.utils.mpi import MPI
from openmdao.devtools.debug import _get_color_printer


_registered = False


def _create_profile_callback(depth, fileset, stream):
    """
    The wrapped function returned from here handles identification of matching calls when called
    as a setprofile callback.
    """
    def _wrapped(frame, event, arg):
        if event == 'call':
            fpath = abspath(frame.f_code.co_filename)
            depth[0] += 1
            if fpath in fileset:
                if 'self' in frame.f_locals:
                    klass = frame.f_locals['self'].__class__.__name__
                else:
                    klass = '?'
                print('c', fpath, frame.f_code.co_firstlineno,
                      frame.f_code.co_name, mem_usage(), time.time(), klass, sep='|', file=stream)
        elif event == 'return' and depth[0] > 0:
            fpath = abspath(frame.f_code.co_filename)
            depth[0] -= 1
            if fpath in fileset:
                print('r', fpath, frame.f_code.co_firstlineno,
                      frame.f_code.co_name, mem_usage(), time.time(), '-', sep='|', file=stream)

    return _wrapped


def _setup(options):
    global _registered, _trace_memory, mem_usage, _out_stream

    if not _registered:
        from openmdao.devtools.memory import mem_usage

        if not options.packages:
            options.packages = ['openmdao']

        # build up the set of all python files that we will (possibly) trace
        fileset = set()
        for p in options.packages:
            fileset.update(_list_package_pyfiles(p))

        if options.outfile is sys.stdout:
            _out_stream = sys.stdout
        else:
            _out_stream = open(options.outfile, "w", buffering=1024*1024)
        _trace_memory = _create_profile_callback(depth=[0], fileset=fileset, stream=_out_stream)
        _registered = True


def setup(**kwargs):
    """
    Setup memory profiling.

    Parameters
    ----------
    kwargs : dict
        Keyword options.
    """
    _setup(_Options(**kwargs))


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


def _list_package_pyfiles(package):
    """
    Return the full path of all python files contained in the given package.
    """
    mod = import_module(package)
    package_path = dirname(abspath(mod.__file__))

    pyfiles = []
    for path, dirlist, filelist in os.walk(package_path):
        if isfile(join(path, '__init__.py')):  # only include valid package dirs
            for f in filelist:
                if f.endswith('.py'):
                    pyfiles.append(join(path, f))

    return pyfiles


def _mem_prof_setup_parser(parser):
    parser.add_argument('-o', '--outfile', action='store', dest='outfile', default='mem_trace.raw',
                        help='Name of file containing memory dump. Default is mem_trace.raw.')
    parser.add_argument('--min', action='store', dest='min_mem', type=float, default=1.0,
                        help='Dump function trace with memory usage in MB above min_mem to the '
                        'given file. Default is 1 MB.')
    parser.add_argument('-c', '--colors', action='store_true', dest='show_colors',
                        help="Display colors if the terminal supports it.  Requires 'colorama' "
                             "python package.  Use 'pip install colorama' to install it.")
    parser.add_argument('--nogc', action='store_true', dest='nogc',
                        help="Disables automatic garbage collection.")
    parser.add_argument('-p', '--package', action='append', dest='packages',
                        default=[], help='Determines which packages will be traced. '
                        'Default package is openmdao.')
    parser.add_argument('-t', '--tree', action='store_true', dest='tree',
                        help="Display memory use in tree format, showing memory use for each "
                             "unique call sequence down to a function.")
    parser.add_argument('file', metavar='file', nargs=1, help='Python file to profile.')


def _mem_prof_exec(options, user_args):
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

    if options.nogc:
        gc.disable()

    # update sys.argv in case python script takes cmd line args
    sys.argv[:] = [progname] + user_args

    exec (code, globals_dict)

    stop()

    _file_line2qualname(options.outfile)

    if options.tree:
        postprocess_memtrace_tree(fname=options.outfile, min_mem=options.min_mem,
                                  show_colors=options.show_colors)
    else:
        postprocess_memtrace_flat(fname=options.outfile, min_mem=options.min_mem,
                                  show_colors=options.show_colors)


@contextmanager
def memtrace(**kwargs):
    """
    Turn on memory tracing within a certain context.

    Parameters
    ----------
    kwargs : dict
        Named options to pass to setup.
    """
    options = _Options(**kwargs)
    if options.outfile is None:
        options.outfile = 'mem_trace.raw'
    if options.min_mem is None:
        options.min_mem = 1.0
    if options.stream is None:
        options.stream = sys.stdout

    _setup(options)
    start()
    yield
    stop()

    _file_line2qualname(options.outfile)
    if options.tree:
        postprocess_memtrace_tree(fname=options.outfile, min_mem=options.min_mem,
                                  show_colors=options.show_colors, stream=options.stream)
    else:
        postprocess_memtrace_flat(fname=options.outfile, min_mem=options.min_mem,
                                  show_colors=options.show_colors, stream=options.stream)


def _mempost_setup_parser(parser):
    parser.add_argument('--out', action='store', dest='outfile', default=None,
                        help='Dump memory report to given file.')
    parser.add_argument('--min', action='store', dest='min_mem', type=float, default=1.0,
                        help='Dump function trace with memory usage to given file.')
    parser.add_argument('-c', '--colors', action='store_true', dest='show_colors',
                        help="Display colors if the terminal supports it.  Requires 'colorama' "
                             "python package.  Use 'pip install colorama' to install it.")
    parser.add_argument('-t', '--tree', action='store_true', dest='tree',
                        help="Display memory use in tree format, showing memory use for each "
                             "unique call sequence down to a function.")
    parser.add_argument('file', metavar='file', nargs=1, help='Memory dump file to process.')


def _mempost_exec(options, user_args):
    """
    Process command line args and perform postprocessing on the specified memory dump file.
    """
    if options.outfile in ('sys.stdout', None):
        stream = sys.stdout
    else:
        stream = open(options.outfile, 'w')

    if options.tree:
        postprocess_memtrace_tree(fname=options.file[0], min_mem=options.min_mem,
                                  show_colors=options.show_colors, stream=stream)
    else:
        postprocess_memtrace_flat(fname=options.file[0], min_mem=options.min_mem,
                                  show_colors=options.show_colors, stream=stream)


def _file_line2qualname(fname):
    # convert the output file to use qualified names immediately.  If we do it later,
    # we risk having mismatches due to changes in source code somewhere between the time we
    # generated the file and the time we postprocess it.
    qual_cache = {}
    inname = '_temp_' + fname

    os.rename(fname, inname)

    with open(inname, "r") as infile:
        with open(fname, 'w') as outfile:

            for line in infile:
                event, fpath, lineno, func, mem, elapsed, klass = line.rstrip().split('|')
                lineno = int(lineno)
                if func[0] == '<':
                    qual = '%s:%s%s' % (fpath, lineno, func)
                else:
                    qual = find_qualified_name(fpath, lineno, qual_cache)
                    qclass = qual.split('.', 1)[0]
                    if klass != '?' and klass != qclass:
                        qual = ':'.join((qual, "(%s)" % klass))
                out_parts = (event, fpath, str(lineno), func, mem, elapsed, qual)
                print('|'.join(out_parts), file=outfile)

    os.remove(inname)


def _process_parts(parts, qual_cache):
    # Convert some strings to int or float.

    event, fpath, lineno, func, mem, elapsed, call = parts
    lineno = int(lineno)
    mem = float(mem)
    elapsed = float(elapsed)

    return [event, fpath, lineno, func, mem, elapsed, call]


def postprocess_memtrace_tree(fname, min_mem=1.0, show_colors=True, rank=0, stream=sys.stdout):
    from openmdao.utils.general_utils import simple_warning
    cprint, Fore, Back, Style = _get_color_printer(stream, show_colors, rank=rank)

    info = {}
    cache = {}

    top = None
    stack = []
    path_stack = []
    qual_cache = {}
    maxmem = 0.0

    if stream is None:
        stream = sys.stdout

    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            parts = _process_parts(line.rstrip().split('|'), qual_cache)
            event, fpath, lineno, func, mem, elapsed, call = parts

            if mem > maxmem:
                maxmem = mem

            if event == 'c':

                path_stack.append(call)
                call_path = '|'.join(path_stack)

                key = call_path

                if key not in info:

                    info[key] = idict = {
                        'calls': 0,
                        'fpath': fpath,
                        'line': lineno,
                        'func': func,
                        'total_mem': 0.,
                        'time': elapsed,
                        'qualname': call,
                        'children': [],
                        'child_ids': set(),
                        'call_path': call_path,
                        'depth': len(path_stack),
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

            elif event == 'r':
                try:
                    c_parts = stack.pop()
                except IndexError:
                    path_stack.pop()
                    continue  # last line is a return from funct called before we start recording

                c_event, c_fpath, c_lineno, c_func, c_mem, c_elapsed, c_call = c_parts

                assert c_fpath == fpath and c_lineno == lineno, "stack error: %s != %s" % (
                    [c_fpath, c_lineno], [fpath, lineno]
                )
                info['|'.join(path_stack)]['total_mem'] += (mem - c_mem)

                path_stack.pop()

    # print out the final results
    seen = set()
    # sort by depth first, then total mem
    for val in sorted(info.values(), key=lambda x: (-x['depth'], x['total_mem']), reverse=True):
        if id(val) in seen:
            continue
        stack = [('', iter([val]), 1.0e20)]
        while stack:
            indent, children, lastmem = stack[-1]
            try:
                val = next(children)
                if id(val) in seen:
                    continue
                seen.add(id(val))
                if val['total_mem'] > min_mem:
                    # if mem is more than the parent, highlight in red
                    if val['total_mem'] > lastmem:
                        fg = Fore.RED
                    else:
                        fg = Fore.GREEN
                    lastmem = val['total_mem']
                    cprint("%s%7.2f " % (indent, val['total_mem']), color=fg + Style.BRIGHT)
                    cprint(" (%d calls)  " % val['calls'])
                    cprint("%s\n" % val['qualname'])
                    stack.append((indent + '   |',
                                 iter(sorted(val['children'], key=lambda x: x['total_mem'],
                                             reverse=True)), lastmem))
            except StopIteration:
                stack.pop()

    cprint("\nMax mem usage: ")
    cprint("%7.2f MB\n" % maxmem, color=Fore.RED + Style.BRIGHT)


def postprocess_memtrace_flat(fname, min_mem=1.0, show_colors=True, rank=0, stream=sys.stdout):
    from openmdao.utils.general_utils import simple_warning
    cprint, Fore, Back, Style = _get_color_printer(stream, show_colors, rank=rank)

    info = {}
    cache = {}

    top = None
    stack = []
    qual_cache = {}
    maxmem = 0.0

    if stream is None:
        stream = sys.stdout

    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            parts = _process_parts(line.rstrip().split('|'), qual_cache)
            event, fpath, lineno, func, mem, elapsed, call = parts

            if mem > maxmem:
                maxmem = mem

            if event == 'c':

                key = call

                if key not in info:

                    info[key] = idict = {
                        'calls': 0,
                        'fpath': fpath,
                        'line': lineno,
                        'func': func,
                        'total_mem': 0.,
                        'time': elapsed,
                        'qualname': call,
                    }
                else:
                    idict = info[key]

                idict['calls'] += 1
                stack.append(parts)

            elif event == 'r':
                try:
                    c_parts = stack.pop()
                except IndexError:
                    continue  # last line is a return from funct called before we start recording

                c_event, c_fpath, c_lineno, c_func, c_mem, c_elapsed, c_call = c_parts

                assert c_fpath == fpath and c_lineno == lineno, "stack error: %s != %s" % (
                    [c_fpath, c_lineno], [fpath, lineno]
                )
                info[c_call]['total_mem'] += (mem - c_mem)

    # print out the final results
    # sort by total mem
    for val in sorted(info.values(), key=lambda x: x['total_mem']):

        if val['total_mem'] > min_mem:
            cprint("%7.2f " % val['total_mem'], color=Fore.GREEN + Style.BRIGHT)
            cprint(" (%d calls)  " % val['calls'])
            cprint("%s\n" % val['qualname'])

    cprint("\nMax mem usage: ")
    cprint("%7.2f MB\n" % maxmem, color=Fore.RED + Style.BRIGHT)
