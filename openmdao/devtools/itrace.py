
import os
import sys
import re
import time
import argparse
import warnings
from contextlib import contextmanager
from collections import defaultdict, OrderedDict

from io import StringIO
from numpy import ndarray
try:
    import objgraph
except ImportError:
    objgraph = None
try:
    import psutil
except ImportError:
    psutil = None

from openmdao.devtools.iprof_utils import _create_profile_callback, find_qualified_name, \
                                         func_group, _collect_methods, _Options, _setup_func_group,\
                                         _get_methods
from openmdao.devtools.memory import mem_usage
from openmdao.utils.mpi import MPI


_trace_calls = None  # pointer to function that implements the trace
_registered = False  # prevents multiple atexit registrations
_printer = None

MAXLINE = 80
tab = '   '
time0 = None

addr_regex = re.compile(" at 0x[0-9a-fA-F]+")

def _indented_print(f_locals, d, indent, excludes=set(['__init__', 'self', '__class__']),
                    show_ptrs=False):
    """
    Print trace info, indenting based on call depth.
    """
    global _printer
    sindent = tab * indent
    sep = '=' if d is f_locals else ':'

    for name in sorted(d, key=lambda a: str(a)):
        if name not in excludes:
            if isinstance(d[name], (dict, OrderedDict)):
                f = StringIO()
                save = _printer
                _printer = _get_printer(f)
                _indented_print(f_locals, d[name], 0, show_ptrs=show_ptrs)
                _printer = save
                s = "  %s%s%s{%s}" % (sindent, name, sep, f.getvalue())
            else:
                s = "  %s%s%s%s" % (sindent, name, sep, d[name])
            if not show_ptrs and ' object at ' in s:
                s = addr_regex.sub('', s)
            linelen = len(s)
            leneq = len(s.split(sep, 1)[0])
            if linelen > MAXLINE:
                if '\n' in s:
                    # change indent
                    s = s.replace("\n", "\n%s" % (' '*leneq))
            _printer(s)


def _get_printer(stream, rank=-1):
    """
    Return a custom print function that outputs to the given stream and on the given rank.
    """
    if MPI and rank >= MPI.COMM_WORLD.size:
        if MPI.COMM_WORLD.rank == 0:
            print("Specified rank (%d) is outside of the valid range (0-%d)." %
                  (rank, MPI.COMM_WORLD.size - 1))
        exit()

    # rank < 0 means output on all ranks
    if not MPI or rank < 0 or MPI.COMM_WORLD.rank == rank:
        def prt(*args, **kwargs):
            print(*args, file=stream, flush=True, **kwargs)
    else:
        def prt(*args, **kwargs):
            pass
    return prt


def _trace_call(frame, arg, stack, context):
    """
    This is called after we have matched based on glob pattern and isinstance check.
    """
    global time0
    if time0 is None:
        time0 = time.time()

    (qual_cache, method_counts, class_counts, id2count,
     verbose, memory, leaks, stream, show_ptrs) = context

    funcname = find_qualified_name(frame.f_code.co_filename,
                                   frame.f_code.co_firstlineno, qual_cache)

    self = frame.f_locals['self']
    try:
        pname = "(%s)" % self.pathname
    except AttributeError:
        pname = ""

    cname = self.__class__.__name__
    my_id = id(self)
    if my_id in id2count:
        id_count = id2count[my_id]
    else:
        class_counts[cname] += 1
        id2count[my_id] = id_count = class_counts[cname]

    sname = "%s#%d%s" % (self.__class__.__name__, id_count, pname)

    fullname = '.'.join((sname, funcname))
    method_counts[fullname] += 1

    indent = tab * (len(stack)-1)
    if verbose:
        _printer("%s--> %s (%d)" % (indent, fullname, method_counts[fullname]))
        _indented_print(frame.f_locals, frame.f_locals, len(stack)-1, show_ptrs=show_ptrs)
    else:
        _printer("%s-->%s" % (indent, fullname))

    if memory is not None:
        memory.append(mem_usage())

    if leaks is not None:
        stats = objgraph.typestats()
        stats['frame'] += 1
        stats['cell'] += 1
        stats['list'] += 1
        leaks.append(stats)


def _trace_return(frame, arg, stack, context):
    """
    This is called when a matched function returns.

    This only happens if show_return is True when setup() is called.
    """
    global time0

    (qual_cache, method_counts, class_counts, id2count,
     verbose, memory, leaks, stream, show_ptrs) = context
    funcname = find_qualified_name(frame.f_code.co_filename,
                                   frame.f_code.co_firstlineno, qual_cache)

    self = frame.f_locals['self']
    try:
        pname = "(%s)" % self.pathname
    except AttributeError:
        pname = ""

    sname = "%s#%d%s" % (self.__class__.__name__, id2count[id(self)], pname)

    indent = tab * len(stack)
    if memory is not None:
        current_mem = mem_usage()
        last_mem = memory.pop()
        if current_mem != last_mem:
            delta = current_mem - last_mem
            _printer("%s<-- %s (time: %8.5f) (total: %6.3f MB) (diff: %+.0f KB)" %
                     (indent, '.'.join((sname, funcname)), time.time() - time0, current_mem,
                     delta * 1024.))

            # add this delta to all callers so when they calculate their own delta, this
            # delta won't be included
            for i in range(len(memory) - 1, -1, -1):
                memory[i] += delta
        else:
            _printer("%s<-- %s (time: %8.5f) (total: %6.3f MB)" %
                     (indent, '.'.join((sname, funcname)), time.time() - time0, current_mem))
    else:
        _printer("%s<-- %s" % (indent, '.'.join((sname, funcname))))

    if verbose:
        if arg is not None:
            s = "%s     %s" % (indent, arg)
            if not show_ptrs and ' object at ' in s:
                s = addr_regex.sub('', s)
            _printer(s)

    if leaks is not None:
        last_objs = leaks.pop()
        for name, _, delta_objs in objgraph.growth(peak_stats=last_objs):
            _printer("%s   %s %+d" % (indent, name, delta_objs))


def _setup(options):
    if not func_group:
        _setup_func_group()

    global _registered, _trace_calls, _printer

    verbose = options.verbose
    memory = options.memory
    leaks = options.leaks

    if not _registered:
        methods = _get_methods(options, default='openmdao')

        call_stack = []
        qual_cache = {}
        method_counts = defaultdict(int)
        class_counts = defaultdict(lambda: -1)
        id2count = {}

        if memory:
            if psutil is None:
                raise RuntimeError("Memory tracing requires the 'psutil' package.  "
                                   "Install it using 'pip install psutil'.")
            memory = []
        else:
            memory = None

        if leaks:
            if objgraph is None:
                raise RuntimeError("Leak detection requires the 'objgraph' package. "
                                   "Install it using 'pip install objgraph'.")
            leaks = []
        else:
            leaks = None

        if options.outfile == 'stdout':
            stream = sys.stdout
        elif options.outfile == 'stderr':
            stream = sys.stderr
        else:
            stream = open(options.outfile, 'w')

        _printer = _get_printer(stream, options.rank)

        _trace_calls = _create_profile_callback(call_stack, _collect_methods(methods),
                                                do_call=_trace_call,
                                                do_ret=_trace_return,
                                                context=(qual_cache, method_counts,
                                                         class_counts, id2count, verbose, memory,
                                                         leaks, stream, options.show_ptrs),
                                                filters=options.filters)


def setup(methods=None, verbose=None, memory=None, leaks=False, rank=-1, show_ptrs=False,
          outfile='stdout'):
    """
    Setup call tracing.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or None
        Methods to be traced, based on glob patterns and isinstance checks.
    verbose : bool
        If True, show function locals and return values.
    memory : bool
        If True, show functions that increase memory usage.
    leaks : bool
        If True, show objects that are created within a function and not garbage collected.
    rank : int
        MPI rank where output is desired.  The default, -1 means output from all ranks.
    show_ptrs : bool
        If True, show addresses of printed objects.
    outfile : file-like or str
        Output file.
    """
    _setup(_Options(methods=methods, verbose=verbose, memory=memory, leaks=leaks, rank=rank,
                    show_ptrs=show_ptrs, outfile=outfile))


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
def tracing(methods=None, verbose=False, memory=False, leaks=False, show_ptrs=False):
    """
    Turn on call tracing within a certain context.

    Parameters
    ----------
    methods : list of (glob, (classes...)) or str or None
        Methods to be traced, based on glob patterns and isinstance checks. If value
        is a string, use that string to lookup a 'canned' method list by name.
    verbose : bool
        If True, show function locals and return values.
    memory : bool
        If True, show functions that increase memory usage.
    leaks : bool
        If True, show objects that are created within a function and not garbage collected.
    show_ptrs : bool
        If True, show addresses of printed objects.
    """
    setup(methods=methods, verbose=verbose, memory=memory, leaks=leaks, show_ptrs=show_ptrs)
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
    memory : bool
        If True, show functions that increase memory usage.
    leaks : bool
        If True, show objects that are created within a function and not garbage collected.
    filters : list of str or None
        If not None, evaluate as an expression in the frame of matching trace functions. If
        True, include the function in the trace.  Up to one expresson per class.
    show_ptrs : bool
        If True, show addresses of printed objects.
    """
    def __init__(self, methods=None, verbose=False, memory=False, leaks=False, filters=None,
                 show_ptrs=False):
        self.options = _Options(methods=methods, verbose=verbose, memory=memory, leaks=leaks,
                                filters=filters, show_ptrs=show_ptrs)
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
    """
    Set up the command line options for the 'openmdao trace' command line tool.
    """
    if not func_group:
        _setup_func_group()

    parser.add_argument('file', nargs=1, help='Python file to be traced.')
    parser.add_argument('-g', '--group', action='store', dest='methods',
                        default='openmdao',
                        help='Determines which group of methods will be traced. Default is "openmdao".'
                              ' Options are: %s' % sorted(func_group.keys()))
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help="Show function locals and return values.")
    parser.add_argument('--ptrs', action='store_true', dest='show_ptrs',
                        help="Show addresses of printed objects.")
    parser.add_argument('-m', '--memory', action='store_true', dest='memory',
                        help="Show memory usage.")
    parser.add_argument('-l', '--leaks', action='store_true', dest='leaks',
                        help="Show objects that are not garbage collected after each function call.")
    parser.add_argument('-r', '--rank', action='store', dest='rank', type=int,
                        default=-1, help='MPI rank where output is desired.  Default is all ranks.')
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        default='stdout', help='Output file.  Defaults to stdout.')
    parser.add_argument('-f', '--filter', action='append', dest='filters',
                        default=[],
                        help='An expression.  If it evaluates to True for any matching trace '
                             'function, that function will be displayed in the trace. One '
                             'expression can be added for each class.')


def _itrace_exec(options, user_args):
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

    sys.argv[:] = [progname] + user_args
    exec (code, globals_dict)
