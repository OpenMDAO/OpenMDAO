from __future__ import print_function

import os
import sys
import ast

from inspect import getmembers
from fnmatch import fnmatchcase
from collections import defaultdict
from six import string_types


class _Options(object):
    """
    A fake options class for use when there is no parser.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return None


class FunctionFinder(ast.NodeVisitor):
    """
    This class locates all of the functions and methods in a file and associates any
    method with its corresponding class.
    """
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
            qual = ("<%s:%d>" % (self.fname, node.lineno), None, node.name)

        self.cache[node.lineno] = qual


def find_qualified_name(filename, line, cache, full=True):
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
    full : bool
        If True, assemble the full name else return the parts

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

            FunctionFinder(filename, fcache).visit(ast.parse(contents, filename))

        cache[filename] = fcache

    if full:
        parts = cache[filename][line]
        if parts[0]:
            return '.'.join((parts[0], parts[2]))
        else:
            return '.'.join((parts[1], parts[2]))

    return cache[filename][line]


# This maps a simple identifier to a group of classes and corresponding
# glob patterns for each class.
func_group = {}


def _setup_func_group():
    global func_group

    from openmdao.core.system import System
    from openmdao.core.problem import Problem
    from openmdao.core.driver import Driver
    from openmdao.solvers.solver import Solver
    from openmdao.jacobians.jacobian import Jacobian
    from openmdao.matrices.matrix import Matrix
    from openmdao.vectors.default_vector import DefaultVector, DefaultTransfer

    func_group.update({
        'openmdao': [
            ("*", (System, Jacobian, Matrix, Solver, Driver, Problem)),
        ],
        'openmdao_all': [
            ("*", (System, DefaultVector, DefaultTransfer, Jacobian, Matrix, Solver, Driver, Problem)),
        ],
        'setup': [
            ("*setup*", (System, Solver, Driver, Problem)),
        ],
        'dataflow': [
            ('*compute*', (System,)),
            ('*linear*', (System,)),
            ('_transfer', (System,)),
            ('*', (DefaultTransfer,)),
        ],
        'linear': [
            ('*linear*', (System,)),
            ('*solve*', (Solver,)),
            ('*compute*', (System,))
        ],
        'solver': [
            ('*', (Solver,))
        ],
        'driver': [
            ('*', (Driver,))
        ],
        'transfer': [
            ('*', (DefaultTransfer,)),
            ('_transfer', (System,))
        ]
    })

    try:
        from mpi4py import MPI
        from petsc4py import PETSc
        from openmdao.vectors.petsc_vector import PETScVector, PETScTransfer

        #TODO: this needs work.  Still lots of MPI calls not covered here...
        func_group['mpi'] = [
            ('*', (PETScTransfer,)),
            ('get_norm', (PETScVector,)),
            ('_initialize_data', (PETScVector,))
        ]
    except ImportError:
        pass


def _collect_methods(method_patterns):
    """
    Iterate over a dict of method name patterns mapped to classes.  Search
    through the classes for anything that matches and return a dict of
    exact name matches and their corresponding classes.

    Parameters
    ----------
    method_patterns : [(pattern1, (class1, class2, ... class_n)), ... (pattern_n, (class_n1, class_n2, ...)]
        List of tuples of glob patterns and lists of classes used for isinstance checks

    Returns
    -------
    defaultdict
        Dict of method names and tuples of all classes that matched for that method. Default value
        of the dict is a class that matches nothing
    """
    matches = defaultdict(list)

    # TODO: update this to also work with stand-alone functions
    for pattern, classes in method_patterns:
        for class_ in classes:
            for name, obj in getmembers(class_):
                if callable(obj) and (pattern == '*' or fnmatchcase(name, pattern)):
                    matches[name].append(class_)

    # convert values to tuples so we can use in isinstance call
    for name in matches:
        lst = matches[name]
        if len(lst) == 1:
            matches[name] = lst[0]
        else:
            matches[name] = tuple(lst)

    return matches


def _create_profile_callback(stack, matches, do_call=None, do_ret=None, context=None):
    """
    The wrapped function returned from here handles identification of matching calls when called
    as a setprofile callback.
    """
    def _wrapped(frame, event, arg):
        if event == 'call':
            if 'self' in frame.f_locals and frame.f_code.co_name in matches and \
                    isinstance(frame.f_locals['self'], matches[frame.f_code.co_name]):
                stack.append(id(frame))
                if do_call is not None:
                   return do_call(frame, arg, stack, context)
        elif event == 'return' and stack:
            if id(frame) == stack[-1]:
                stack.pop()
                if do_ret is not None:
                    do_ret(frame, arg, stack, context)
                #stack.pop()

    return _wrapped


def _get_methods(options, default):
    if options.methods is None:
        methods = func_group[default]
    elif isinstance(options.methods, string_types):
        try:
            methods = func_group[options.methods]
        except KeyError:
            raise KeyError("Unknown function group '%s'." % options.methods)
    else:
        methods = options.methods

    return methods
