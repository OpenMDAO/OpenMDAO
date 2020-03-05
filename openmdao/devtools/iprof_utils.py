
import os
import sys
import ast

from inspect import getmembers
from fnmatch import fnmatchcase
from collections import defaultdict


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
        self.stack = []

    def _do_callable_def(self, node):
        if self.stack:
            qual =  (None, '.'.join(self.stack),  node.name)
        else:
            qual = ("<%s:%d>" % (self.fname, node.lineno), None, node.name)
        self.cache[node.lineno] = qual
        # some versions of python report different line numbers for funnctions/classes with
        # decorators, so just put keys in the cache dict for all of the decorator line numbers
        # as well in order to avoid any KeyErrors.
        for d in node.decorator_list:
            self.cache[d.lineno] = qual

        self.stack.append(node.name)
        for bnode in node.body:
            self.visit(bnode)
        self.stack.pop()

    def visit_ClassDef(self, node):
        self._do_callable_def(node)

    def visit_FunctionDef(self, node):
        self._do_callable_def(node)


def find_qualified_name(filename, line, cache, full=True):
    """
    Determine full function name (class.method) or function for unbound functions.

    Parameters
    ----------
    filename : str
        Name of file containing source code.
    line : int
        Line number within the given file.
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

base_classes = {}


def _setup_func_group():
    global func_group, base_classes

    from openmdao.core.system import System
    from openmdao.core.component import Component
    from openmdao.core.explicitcomponent import ExplicitComponent
    from openmdao.core.problem import Problem
    from openmdao.core.driver import Driver
    from openmdao.core.total_jac import _TotalJacInfo
    from openmdao.solvers.solver import Solver, LinearSolver
    from openmdao.solvers.nonlinear.newton import NewtonSolver
    from openmdao.solvers.linear.direct import DirectSolver
    from openmdao.jacobians.jacobian import Jacobian
    from openmdao.matrices.matrix import Matrix
    from openmdao.vectors.default_vector import DefaultVector, DefaultTransfer
    from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme

    for class_ in [System, ExplicitComponent, Problem, Driver, _TotalJacInfo, Solver, LinearSolver,
                   NewtonSolver, Jacobian, Matrix, DefaultVector, DefaultTransfer]:
        base_classes[class_.__name__] = class_


    func_group.update({
        'openmdao': [
            ("*", (System, Jacobian, Matrix, Solver, Driver, Problem)),
        ],
        'openmdao_all': [
            ("*", (System, DefaultVector, DefaultTransfer, Jacobian, Matrix, Solver, Driver,
                   Problem, _TotalJacInfo)),
        ],
        'setup': [
            ("__init__", (System, Solver, Driver, Problem, Jacobian, DefaultVector, _TotalJacInfo,
                          Matrix)),
            ("*setup*", (System, Solver, Driver, Problem, Jacobian, DefaultVector, _TotalJacInfo,
                         Matrix)),
            ('_configure', (System,)),
            ('set_initial_values', (System,)),
            ('_set_initial_conditions', (Problem,)),
            ('_build', (Matrix,)),
            ('_add_submat', (Matrix,)),
            ('_get_maps', (System,)),
            ('_set_approx_partials_meta', (System,)),
            ('_init_relevance', (System,)),
            ('_get_initial_*', (System,)),
            ('_initialize_*', (DefaultVector,)),
            ('_create_*', (DefaultVector,)),
            ('_extract_root_data', (DefaultVector,)),
        ],
        'dataflow': [
            ('*compute*', (System,)),
            ('*linear*', (System,)),
            ('_transfer', (System,)),
            ('*', (DefaultTransfer,)),
        ],
        'linear': [
            ('_apply_linear', (System,)),
            ('_setup_jacobians', (System, Solver)),
            ('_solve_linear', (System,)),
            ('apply_linear', (System,)),
            ('solve_linear', (System,)),
            ('_set_approx_partials_meta', (System, Jacobian)),
            ('_linearize', (System, Solver)),
            # include NewtonSolver to provide some context
            ('solve', (LinearSolver, NewtonSolver)),
            ('_update', (Jacobian,)),
            ('_apply', (Jacobian,)),
            ('_initialize', (Jacobian,)),
            ('compute_totals', (_TotalJacInfo, Problem, Driver)),
            ('compute_totals_approx', (_TotalJacInfo,)),
            ('compute_jacvec_product', (System,)),
        ],
        'jac': [
            ('_linearize', (System, DirectSolver)),
            ('_setup_jacobians', (System,)),
            ('compute_totals', (_TotalJacInfo, Problem, Driver)),
            ('compute_totals_approx', (_TotalJacInfo,)),
            ('_apply_linear', (System,)),
            ('solve', (LinearSolver, NewtonSolver)),
            ('_update', (Jacobian,)),
            ('_initialize', (Jacobian,)),
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
        ],
        'coloring': [
            ('*_approx_*', (Driver, System)),
            ('*color*', (object,)),
            ('*partials*', (System, Driver)),
            ('*sparsity*', (Jacobian, System, Driver)),
            ('*simul*', (Driver, System)),
            ('*jacobian*', (Driver, System)),
            ('_setup', (System,)),
            ('_final_setup', (System,)),
        ]
        # NOTE: context managers and other functions that yield instead of return will NOT show
        # up properly in the trace.  For example, our context managers for scaling will show up
        # as a call and immediate return from the context manager, followed by the functions that
        # should show up as inside of the context manager but don't.  This is just here to
        # remind me not to try to create a 'scaling' group again.
        # 'scaling': [
        #     ('*scaled_context*', (System,)),
        #     ('compute*', (Component, ApproximationScheme)),
        #     ('_solve*', (System,)),
        #     ('solve_*', (System,)),
        #     ('run_*', (System,)),
        #     ('guess_*', (System,)),
        #     ('*apply*', (System,)),
        #     ('_apply', (Jacobian,)),
        #     ('*linearize', (System,)),
        # ],
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


def _collect_methods(method_patterns=None):
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
    if method_patterns is None:
        return None

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


def _create_profile_callback(stack, matches=None, do_call=None, do_ret=None, context=None,
                             filters=None):
    """
    The wrapped function returned from here handles identification of matching calls when called
    as a setprofile callback.
    """
    if filters:
        newfilts = []
        for s in filters:
            class_name, filt = s.split(' ', 1)
            class_ = base_classes[class_name]
            newfilts.append((class_, compile(filt, mode='eval', filename=filt)))
        filters = newfilts

    def _wrapped(frame, event, arg):
        if event == 'call':
            if matches is None:
                stack.append(id(frame))
                if do_call is not None:
                    return do_call(frame, arg, stack, context)
            elif 'self' in frame.f_locals and frame.f_code.co_name in matches and \
                    isinstance(frame.f_locals['self'], matches[frame.f_code.co_name]):
                pred = True
                if filters:
                    inst = frame.f_locals['self']
                    for class_, filt in filters:
                        if isinstance(inst, class_):
                            pred = eval(filt, globals(), frame.f_locals)
                            break
                if pred:
                    stack.append(id(frame))
                    if do_call is not None:
                       return do_call(frame, arg, stack, context)
        elif event == 'return' and stack:
            if id(frame) == stack[-1]:
                stack.pop()
                if do_ret is not None:
                    do_ret(frame, arg, stack, context)

    return _wrapped


def _get_methods(options, default):
    if options.methods is None:
        methods = func_group[default]
    elif isinstance(options.methods, str):
        try:
            methods = func_group[options.methods]
        except KeyError:
            raise KeyError("Unknown function group '%s'." % options.methods)
    else:
        methods = options.methods

    return methods
