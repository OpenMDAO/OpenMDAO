from __future__ import print_function

import os
import sys
import ast

from inspect import getmembers
from fnmatch import fnmatchcase
from contextlib import contextmanager

from six import iteritems, PY3

from openmdao.core.system import System
from openmdao.core.problem import Problem
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver
from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.matrix import Matrix
from openmdao.vectors.vector import Vector, Transfer


class ClassVisitor(ast.NodeVisitor):
    def __init__(self, fname, cache):
        ast.NodeVisitor.__init__(self)
        self.fname = fname
        self.cache = cache
        self.class_stack = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        for bnode in node.body:
            self.visit(bnode)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        if self.class_stack:
            qual =  (None, '.'.join(self.class_stack),  node.name)
        else:
            qual = ("<%s>" % self.fname, None, node.name)

        self.cache[node.lineno] = qual


def find_qualified_name(filename, line, cache):
    """
    Determine full function name (class.method) or function for unbound functions.

    Parameters
    ----------
    filename : str
        Name of file containing source code.
    line : int
        Line number within the give file.
    cache : dict
        A dictionary containing infomation by filename.

    Returns
    -------
    str or None
        Fully qualified function/method name or None.
    """

    if filename not in cache:
        fcache = {}

        with open(filename, 'Ur') as f:
            contents = f.read()
            if len(contents) > 0 and contents[-1] != '\n':
                contents += '\n'

            ClassVisitor(filename, fcache).visit(ast.parse(contents, filename))

        cache[filename] = fcache

    return cache[filename][line]


# This maps a simple identifier to a group of classes and corresponding
# glob patterns for each class.
_trace_dict = {
    'openmdao': [
        ("*", (System, Jacobian, Matrix, Solver, Driver, Problem)),
    ],
    'memory': [
        ("*", (System, Vector, Jacobian, Matrix, Solver, Driver, Problem)),
    ],
    'setup': [
        ("*setup*", (System, Solver, Driver, Problem)),
    ],
}


def func_group(name):
    return _trace_dict[name]


# final dict of method names and their corresponding classes
_matches = {}
_method_counts = {}
_mem_changes = {}
_callstack = []
_trace_callback = None  # pointer to function that implements the trace
_registered = False  # prevents multiple atexit registrations
_stack = []
_qual_cache = {}  # cache of files scanned for qualified names
_memstack = []


def _create_trace_func(do_call=None, do_ret=None):
    """
    The wrapped function returned from here handles identification of matching calls.
    """
    def _wrapped(frame, event, arg):
        if event == 'call':
            if frame.f_code.co_name in _matches and 'self' in frame.f_locals and  \
                    isinstance(frame.f_locals['self'], _matches[frame.f_code.co_name]):
                if do_call is not None:
                    do_call(frame, arg)
                _stack.append(frame)

        elif event == 'return' and _stack:
            if frame is _stack[-1]:
                if do_ret is not None:
                    do_ret(frame, arg)
                _stack.pop()

    return _wrapped


def _trace_call(frame, arg):
    qfile, qclass, qname = find_qualified_name(frame.f_code.co_filename,
                                               frame.f_code.co_firstlineno,
                                               _qual_cache)
    if qfile is None:
        fullname = '.'.join((qclass, qname))
    else:
        fullname = ':'.join((qfile, qname))

    if fullname in _method_counts:
        _method_counts[fullname] += 1
    else:
        _method_counts[fullname] = 1
    print('   ' * len(_stack),
          "%s (%d)" % (fullname, _method_counts[fullname]))



_trace_calls = _create_trace_func(_trace_call)

def _trace_mem_call(frame, arg):
    _memstack.append((frame.f_locals['self'], frame.f_code, mem_usage()))

def _trace_mem_ret(frame, arg):
    _, code_obj, mem_start = _memstack.pop()
    delta = mem_usage() - mem_start
    if delta > 0.0:
        if code_obj in _mem_changes:
            _mem_changes[(frame.f_locals['self'], code_obj)] += delta
        else:
            _mem_changes[(frame.f_locals['self'], code_obj)] = delta

        # we only want to see deltas from the routines that actually allocate
        # memory rather then those routines and all of the routines that call
        # them either directly or indirectly, so we add the current delta to
        # the mem usage up the call stack, which will subtract it from the ancestor
        # deltas.
        for i in range(len(_callstack)):
            _memstack[i][2] += delta


_trace_memory = _create_trace_func(_trace_mem_call, _trace_mem_ret)


def _collect_methods(method_patterns):
    """
    Iterate over a dict of method name patterns mapped to classes.  Search
    through the classes for anything that matches and return a dict of
    exact name matches and their correspoding classes.

    Parameters
    ----------
    method_patterns : [(pattern1, (class1, class2, ... class_n)), ... (pattern_n, (class_n1, class_n2, ...)]
        List of tuples of glob patterns and lists of classes used for isinstance checks

    Returns
    -------
    dict
        Dict of method names and tuples of all classes that matched for that method.
    """
    matches = {}

    # TODO: update this to also work with stand-alone functions
    for pattern, classes in method_patterns:
        for class_ in classes:
            for name, obj in getmembers(class_):
                if callable(obj) and (pattern == '*' or fnmatchcase(name, pattern)):
                    if name in matches:
                        matches[name].append(class_)
                    else:
                        matches[name] = [class_]

    # convert values to tuples so we can use in isinstance call
    for name in matches:
        matches[name] = tuple(matches[name])

    return matches


def setup(trace_type='call', methods=None):
    global _registered, _matches, _file2class, _trace_callback
    if not _registered:
        if methods is None:
            methods = func_group('openmdao')

        _matches = _collect_methods(methods)

        if trace_type == 'mem':
            global mem_usage
            from openmdao.devtools.debug import mem_usage

            _trace_callback = _trace_memory

            def print_totals():
                items = sorted(_mem_changes.items(), key=lambda x: x[1])
                for (self, code_obj), delta in items:
                    if delta != 0.0:
                        qfile, qclass, qname = find_qualified_name(code_obj.co_filename,
                                                                   code_obj.co_firstlineno,
                                                                   _qual_cache)
                        if qfile is None:
                            fullname = '.'.join((qclass, qname))
                        else:
                            fullname = ':'.join((qfile, qname))
                        print("%s %g MB" % (fullname, delta))
            import atexit
            atexit.register(print_totals)
            _registered = True
        else:
            _trace_callback = _trace_calls


def start():
    global _matches, _trace_callback
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    if _trace_callback is None:
        raise RuntimeError("trace.setup() was not called before trace.start().")
    sys.setprofile(_trace_callback)


def stop():
    sys.setprofile(None)


@contextmanager
def tracing(trace_type='call', methods=None):
    setup(trace_type, methods=methods)
    start()
    yield
    stop()


class tracedfunc(object):
    """
    Decorator that activates tracing for a particular function.

    Parameters
    ----------
    trace_type : str, optional
        Type of tracing to perform.  Options are ['call', 'mem']
    trace_group : str, optional
        Identifier of a group of classes and methods to trace.
    """
    def __init__(self, trace_type='call', methods=None):
        self.trace_type = trace_type
        self.methods = methods

    def __call__(self, func):
        setup(trace_type=self.trace_type, methods=self.methods)

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
    parser.add_option('-t', '--type', dest="type_",
        help="Type of trace (print calls or trace mem usage)", default='call')
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

        setup(trace_type=options.type_, methods=func_group(options.group))
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
