"""
Utilities for the use of jax in combination with OpenMDAO.
"""
import os
import ast
import textwrap
import inspect
import weakref
from itertools import chain
from collections import defaultdict

import networkx as nx
import numpy as np
from numpy import ndarray

from openmdao.visualization.tables.table_builder import generate_table
from openmdao.utils.code_utils import _get_long_name, replace_method, get_partials_deps


def jit_stub(f, *args, **kwargs):
    """
    Provide a dummy jit decorator for use if jax is not available.

    Parameters
    ----------
    f : Callable
        The function or method to be wrapped.
    *args : list
        Positional arguments.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    Callable
        The decorated function.
    """
    return f


try:
    import jax
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
    import jax.numpy as jnp
    from jax import jit, tree_util, Array
except ImportError:

    jax = None
    jnp = np
    jit = jit_stub


def register_jax_component(comp_class):
    """
    Provide a class decorator that registers the given class as a pytree_node.

    This allows jax to use jit compilation on the methods of this class if they
    reference attributes of the class itself, such as `self.options`.

    Note that this decorator is not necessary if the given class does not reference
    `self` in any methods to which `jax.jit` is applied.

    Parameters
    ----------
    comp_class : class
        The decorated class.

    Returns
    -------
    object
        The same class given as an argument.

    Raises
    ------
    NotImplementedError
        If this class does not define the `_tree_flatten` and _tree_unflatten` methods.
    RuntimeError
        If jax is not available.
    """
    if jax is None:
        raise RuntimeError("jax is not available. "
                           "Try 'pip install openmdao[jax]' with Python>=3.8.")

    if not hasattr(comp_class, '_tree_flatten'):
        raise NotImplementedError(f'class {comp_class} does not implement method _tree_flatten.'
                                  f'\nCannot register {comp_class} as a jax jit-compatible '
                                  f'component.')

    if not hasattr(comp_class, '_tree_unflatten'):
        raise NotImplementedError(f'class {comp_class} does not implement method _tree_unflatten.'
                                  f'\nCannot register class {comp_class} as a jax jit-compatible '
                                  f'component.')

    jax.tree_util.register_pytree_node(comp_class,
                                       comp_class._tree_flatten,
                                       comp_class._tree_unflatten)
    return comp_class


def dump_jaxpr(closed_jaxpr):
    """
    Print out the contents of a Jaxpr.

    Parameters
    ----------
    closed_jaxpr : jax.core.ClosedJaxpr
        The Jaxpr to be examined.
    """
    jaxpr = closed_jaxpr.jaxpr
    print("invars:", jaxpr.invars)
    print("in_avals", closed_jaxpr.in_avals, closed_jaxpr.in_avals[0].dtype)
    print("outvars:", jaxpr.outvars)
    print("out_avals:", closed_jaxpr.out_avals)
    print("constvars:", jaxpr.constvars)
    for eqn in jaxpr.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
    print()
    print("jaxpr:", jaxpr)


def get_func_graph(func, *args, **kwargs):
    """
    Generate a networkx graph from a jax function.

    Parameters
    ----------
    func : Callable
        The function to be analyzed.
    *args : list
        Positional arguments.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    networkx.DiGraph
        The graph representing the function.
    """
    closed_jaxpr = jax.make_jaxpr(func)(*args, **kwargs)
    jaxpr = closed_jaxpr.jaxpr

    graph = nx.DiGraph()
    for i, name in enumerate(chain(jaxpr.invars, jaxpr.outvars)):
        graph.add_node(i, label=f"({i}){name}")  # use index to map to real varnames later
    n2index = {str(n): i for i, n in enumerate(chain(jaxpr.invars, jaxpr.outvars))}
    i = len(n2index)

    for eqn in jaxpr.eqns:
        for inp in eqn.invars:
            if type(inp) is jax._src.core.Var:
                inp = str(inp)
                if inp not in n2index:
                    n2index[inp] = i
                    graph.add_node(i, label=inp)
                    i += 1
                for out in eqn.outvars:
                    if type(out) is jax._src.core.Var:
                        out = str(out)
                        if out not in n2index:
                            n2index[out] = i
                            graph.add_node(i, label=out)
                            i += 1
                        graph.add_edge(n2index[inp], n2index[out])

    # show the function graph visually
    # from openmdao.visualization.graph_viewer import write_graph, _to_pydot_graph
    # G = _to_pydot_graph(graph)
    # write_graph(G)

    return graph


class CompJaxifyBase(ast.NodeTransformer):
    """
    An ast.NodeTransformer that transforms a function definition to jax compatible form.

    So original func becomes compute_primal(self, arg1, arg2, ...).

    If the component has discrete inputs, they will be passed individually into compute_primal
    *before* the continuous inputs.  If the component has discrete outputs, they will be assigned
    to local variables of the same name within the function and set back into the discrete
    outputs dict just prior to the return from the function.

    Parameters
    ----------
    comp : Component
        The Component whose function is to be transformed. This NodeTransformer may only
        be used after the Component has had its _setup_var_data method called, because that
        determines the ordering of the inputs and outputs.
    funcname : str
        The name of the function to be transformed.
    verbose : bool
        If True, the transformed function will be printed to stdout.

    Attributes
    ----------
    _comp : weakref.ref
        A weak reference to the Component whose function is being transformed.
    _funcname : str
        The name of the function being transformed.
    _compute_primal_args : list
        The argument names of the compute_primal function.
    _compute_primal_returns : list
        The names of the outputs from the new function.
    compute_primal : function
        The compute_primal function created from the original function.
    _orig_args : list
        The original argument names of the original function.
    _new_ast : ast node
        The new ast node created from the original function.
    get_self_statics : function
        A function that returns the static args for the Component as a single tuple.
    """

    # these ops require static objects so their args should not be traced.  Traced array ops should
    # use jnp and static ones should use np.
    _static_ops = {'reshape'}
    _np_names = {'np', 'numpy'}

    def __init__(self, comp, funcname, verbose=False):  # noqa
        self._comp = weakref.ref(comp)
        self._funcname = funcname
        func = getattr(comp, funcname)
        if 'jnp' not in func.__globals__:
            func.__globals__['jnp'] = jnp
        namespace = func.__globals__.copy()

        static_attrs, static_dcts = get_self_static_attrs(func)
        self_statics = ['_self_statics_'] if static_attrs or static_dcts else []
        if self_statics:
            self.get_self_statics = self._get_self_statics_func(static_attrs, static_dcts)
        else:
            self.get_self_statics = None

        self._orig_args = list(inspect.signature(func).parameters)

        # ensure that ordering of args and returns exactly matches the order of the inputs and
        # outputs vectors.
        self._compute_primal_args = self._get_compute_primal_args()
        self._compute_primal_returns = self._get_compute_primal_returns()

        node = self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec'))
        self._new_ast = ast.fix_missing_locations(node)

        code = compile(self._new_ast, '<ast>', 'exec')
        exec(code, namespace)    # nosec
        self.compute_primal = namespace['compute_primal']

        if verbose:
            print(f"\n{comp.pathname}:\n{self.get_compute_primal_src()}\n")

    def get_compute_primal_src(self):
        """
        Return the source code of the transformed function.

        Returns
        -------
        str
            The source code of the transformed function.
        """
        return ast.unparse(self._new_ast)

    def get_class_src(self):
        """
        Return the source code of the class containing the transformed function.

        Returns
        -------
        str
            The source code of the class containing the transformed function.
        """
        return replace_method(self._comp().__class__, self._funcname, self.get_compute_primal_src())

    def compute_dependency_iter(self):
        """
        Get (output, input) pairs where the output depends on the input for the compute function.

        Yields
        ------
        tuple
            A tuple of the form (output, input), where output depends on the input.
        """
        yield from get_partials_deps(self.compute_primal)

    def _get_self_statics_func(self, static_attrs, static_dcts):
        fsrc = ['def get_self_statics(self):']
        tupargs = []
        for attr in static_attrs:
            tupargs.append(f"self.{attr}")
        for name, entries in static_dcts:
            for entry in entries:
                tupargs.append(f"self.{name}['{entry}']")
            if len(entries) == 1:
                tupargs.append('')  # so we'll get a trailing comma for a 1 item tuple
        fsrc.append(f'    return ({", ".join(tupargs)})')
        fsrc = '\n'.join(fsrc)
        namespace = getattr(self._comp(), self._funcname).__globals__.copy()
        exec(fsrc, namespace)  # nosec
        return namespace['get_self_statics']

    def _get_pre_body(self):
        if not self._comp()._discrete_outputs:
            return []

        # add a statement to pull individual values out of the discrete outputs
        elts = [ast.Name(id=name, ctx=ast.Store()) for name in self._comp()._discrete_outputs]
        return [
            ast.Assign(targets=[ast.Tuple(elts=elts, ctx=ast.Store())],
                       value=ast.Call(
                           func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='self',
                                                                                 ctx=ast.Load()),
                                                                  attr='_discrete_outputs',
                                                                  ctx=ast.Load()),
                                              attr='values', ctx=ast.Load()),
                                      args=[], keywords=[]))]

    def _get_post_body(self):
        if not self._comp()._discrete_outputs:
            return []

        # add a statement to set the values of self._discrete outputs
        elts = [ast.Name(id=name, ctx=ast.Load()) for name in self._comp()._discrete_outputs]
        args = [ast.Tuple(elts=elts, ctx=ast.Load())]
        return [ast.Expr(value=ast.Call(func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                attr='_discrete_outputs', ctx=ast.Load()),
                                attr='set_vals', ctx=ast.Load()), args=args, keywords=[]))]

    def _make_return(self):
        val = ast.Tuple([ast.Name(id=n, ctx=ast.Load()) for n in self._compute_primal_returns],
                        ctx=ast.Load())
        return ast.Return(val)

    def _get_new_args(self):
        new_args = [ast.arg('self', annotation=None)]
        for arg_name in self._compute_primal_args:
            new_args.append(ast.arg(arg=arg_name, annotation=None))
        return ast.arguments(args=new_args, posonlyargs=[], vararg=None, kwonlyargs=[],
                             kw_defaults=[], kwarg=None, defaults=[])

    def visit_FunctionDef(self, node):
        """
        Transform the compute function definition.

        The function will be transformed from compute(self, inputs, outputs) to
        compute_primal(self, arg1, arg2, ...) where args are the input values in the order they are
        stored in inputs.  All subscript accesses into the input args will be replaced with the name
        of the key being accessed, e.g., inputs['foo'] becomes foo. The new function will return a
        tuple of the output values in the order they are stored in outputs.  If compute has the
        additional args discrete_inputs and discrete_outputs, they will be handled similarly.

        Parameters
        ----------
        node : ast.FunctionDef
            The FunctionDef node being visited.

        Returns
        -------
        ast.FunctionDef
            The transformed node.
        """
        newbody = self._get_pre_body()
        newbody.extend([self.visit(statement) for statement in node.body])
        newbody.extend(self._get_post_body())
        # add a return statement for the outputs
        newbody.append(self._make_return())

        newargs = self._get_new_args()
        return ast.FunctionDef('compute_primal', newargs, newbody, node.decorator_list,
                               node.returns, node.type_comment)

    def visit_Subscript(self, node):
        """
        Translate a Subscript node into a Name node with the name of the subscript variable.

        Parameters
        ----------
        node : ast.Subscript
            The Subscript node being visited.

        Returns
        -------
        ast.Any
            The transformed node.
        """
        # if we encounter a subscript of any of the input args, then replace arg['name'] or
        # arg["name"] with name.

        # NOTE: this will only work if the subscript is a string constant. If the subscript is a
        # variable or some other expression, then we don't modify it.
        if (isinstance(node.value, ast.Name) and node.value.id in self._orig_args and
                isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            return ast.copy_location(ast.Name(id=_fixname(node.slice.value), ctx=node.ctx), node)

        return self.generic_visit(node)

    def visit_Attribute(self, node):
        """
        Translate any non-static use of 'numpy' or 'np' to 'jnp'.

        Parameters
        ----------
        node : ast.Attribute
            The Attribute node being visited.

        Returns
        -------
        ast.Any
            The transformed node.
        """
        if isinstance(node.value, ast.Name) and node.value.id in self._np_names:
            if node.attr not in self._static_ops:
                return ast.copy_location(ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()),
                                                       attr=node.attr, ctx=node.ctx), node)

        return self.generic_visit(node)


class ExplicitCompJaxify(CompJaxifyBase):
    """
    An ast.NodeTransformer that transforms a compute function definition to jax compatible form.

    So compute(self, inputs, outputs) becomes compute_primal(self, arg1, arg2, ...) where args are
    the input values in the order they are stored in inputs.  The new function will return a tuple
    of the output values in the order they are stored in outputs.

    If the component has discrete inputs, they will be passed individually into compute_primal
    *before* the continuous inputs.  If the component has discrete outputs, they will be assigned
    to local variables of the same name within the function and set back into the discrete
    outputs dict just prior to the return from the function.

    Parameters
    ----------
    comp : ExplicitComponent
        The Component whose compute function is to be transformed. This NodeTransformer may only
        be used after the Component has had its _setup_var_data method called, because that
        determines the ordering of the inputs and outputs.
    verbose : bool
        If True, the transformed function will be printed to stdout.
    """

    def __init__(self, comp, verbose=False):  # noqa
        super().__init__(comp, 'compute', verbose)

    def _get_arg_values(self):
        discrete_inputs = self._comp()._discrete_inputs
        yield from discrete_inputs.values()
        if self._comp()._inputs is None:
            for name, meta in self._comp()._var_rel2meta['input'].items():
                if name not in discrete_inputs:
                    yield meta['value']
        else:
            yield from self._comp()._inputs.values()

    def _get_compute_primal_args(self):
        # ensure that ordering of args and returns exactly matches the order of the inputs and
        # outputs vectors.
        return [n for n in chain(self._comp()._discrete_inputs,
                                 self._comp()._var_rel_names['input'])]

    def _get_compute_primal_returns(self):
        return [n for n in chain(self._comp()._discrete_outputs,
                                 self._comp()._var_rel_names['output'])]


class ImplicitCompJaxify(CompJaxifyBase):
    """
    A NodeTransformer that transforms an apply_nonlinear function definition to jax compatible form.

    So apply_nonlinear(self, inputs, outputs, residuals) becomes
    compute_primal(self, arg1, arg2, ...) where args are
    the input and output values in the order they are stored in their respective Vectors.
    The new function will return a tuple of the residual values in the order they are stored in
    the residuals Vector.

    If the component has discrete inputs, they will be passed individually into compute_primal
    *before* the continuous inputs.  If the component has discrete outputs, they will be assigned
    to local variables of the same name within the function and set back into the discrete
    outputs dict just prior to the return from the function.

    Parameters
    ----------
    comp : ImplicitComponent
        The Component whose apply_nonlinear function is to be transformed. This NodeTransformer
        may only be used after the Component has had its _setup_var_data method called, because that
        determines the ordering of the inputs, outputs, and residuals.
    verbose : bool
        If True, the transformed function will be printed to stdout.
    """

    def __init__(self, comp, verbose=False):  # noqa
        super().__init__(comp, 'apply_nonlinear', verbose)

    def _get_arg_values(self):
        discrete_inputs = self._comp()._discrete_inputs
        yield from discrete_inputs.values()

        comp = self._comp()
        if comp._inputs is None:
            for name, meta in comp._var_rel2meta['input'].items():
                if name not in discrete_inputs:
                    yield meta['value']
        else:
            yield from comp._inputs.values()

        if comp._outputs is None:
            for name, meta in comp._var_rel2meta['output'].items():
                if name not in comp._discrete_outputs:
                    yield meta['value']
        else:
            yield from comp._outputs.values()

    def _get_compute_primal_args(self):
        # ensure that ordering of args and returns exactly matches the order of the inputs,
        # outputs, and residuals vectors.
        return [n for n in chain(self._comp()._discrete_inputs,
                                 self._comp()._var_rel_names['input'],
                                 self._comp()._var_rel_names['output'])]

    def _get_compute_primal_returns(self):
        return [n for n in chain(self._comp()._discrete_outputs,
                                 self._comp()._var_rel_names['output'])]


class SelfAttrFinder(ast.NodeVisitor):
    """
    An ast.NodeVisitor that collects all attribute names that are accessed on `self`.

    Parameters
    ----------
    method : method
        The method to be analyzed.

    Attributes
    ----------
    _attrs : set
        The set of attribute names accessed on `self`.
    _funcs : set
        The set of method names accessed on `self`.
    _dcts : dict
        The set of attribute names accessed on `self` that are subscripted.
    """

    # TODO: need to support intermediate variables, e.g., foo = self.options,   x = foo['blah']
    # TODO: need to support self.options[var], where var is an attr, not a string.
    # TODO: even if we can't handle the above, at least detect and flag them and warn that
    #       auto-converter can't handle them.
    def __init__(self, method):  # noqa
        self._attrs = set()
        self._funcs = set()
        self._dcts = defaultdict(set)
        self.visit(ast.parse(textwrap.dedent(inspect.getsource(method)), mode='exec'))

    def visit_Attribute(self, node):
        """
        Visit an Attribute node.

        If the attribute is accessed on `self`, add the attribute name to the set of attributes.

        Parameters
        ----------
        node : ast.Attribute
            The Attribute node being visited.
        """
        name = _get_long_name(node)
        if name is None:
            return
        if name.startswith('self.'):
            self._attrs.add(name.partition('.')[2])

    def visit_Subscript(self, node):
        """
        Visit a Subscript node.

        If the subscript is accessed on `self`, add the attribute name to the set of attributes.

        Parameters
        ----------
        node : ast.Subscript
            The Subscript node being visited.
        """
        name = _get_long_name(node.value)
        if name is None:
            return
        if name.startswith('self.'):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                self._dcts[name.partition('.')[2]].add(node.slice.value)
            else:
                self._attrs.add(name.partition('.')[2])
        self.visit(node.slice)

    def visit_Call(self, node):
        """
        Visit a Call node.

        If the function is accessed on `self`, add the function name to the set of functions.

        Parameters
        ----------
        node : ast.Call
            The Call node being visited.
        """
        name = _get_long_name(node.func)
        if name is not None and name.startswith('self.'):
            parts = name.split('.')
            if len(parts) == 2:
                self._funcs.add(parts[1])
            else:
                self._attrs.add('.'.join(parts[1:-1]))

        for arg in node.args:
            self.visit(arg)


def get_self_static_attrs(method):
    """
    Get the set of attribute names accessed on `self` in the given method.

    Parameters
    ----------
    method : method
        The method to be analyzed.

    Returns
    -------
    set
        The set of attribute names accessed on `self`.
    dict
        The set of attribute names accessed on `self` that are subscripted with a string.
    """
    saf = SelfAttrFinder(method)
    static_attrs = sorted(saf._attrs)
    static_dcts = [(name, sorted(eset)) for name, eset in sorted(saf._dcts.items(),
                                                                 key=lambda x: x[0])]

    return static_attrs, static_dcts


_invalid = frozenset((':', '(', ')', '[', ']', '{', '}', ' ', '-',
                      '+', '*', '/', '^', '%', '!', '<', '>', '='))


def _fixname(name):
    """
    Convert (if necessary) the given name into a valid Python variable name.

    Parameters
    ----------
    name : str
        The name to be fixed.

    Returns
    -------
    str
        The fixed name.
    """
    intr = _invalid.intersection(name)
    if intr:
        for c in intr:
            name = name.replace(c, '_')
    return name


def benchmark_component(comp_class, methods=(None, 'cs', 'jax'), initial_vals=None, repeats=2,
                        mode='auto', table_format='simple_grid', **kwargs):
    """
    Benchmark the performance of a Component using different methods for computing derivatives.

    Parameters
    ----------
    comp_class : class
        The class of the Component to be benchmarked.
    methods : tuple of str
        The methods to be benchmarked. Options are 'cs', 'jax', and None.
    initial_vals : dict or None
        Initial values for the input variables.
    repeats : int
        The number of times to run compute/compute_partials.
    mode : str
        The preferred derivative direction for the Problem.
    table_format : str or None
        If not None, the format of the table to be displayed.
    **kwargs : dict
        Additional keyword arguments to be passed to the Component.

    Returns
    -------
    dict
        A dictionary containing the benchmark results.
    """
    import time

    from openmdao.core.problem import Problem
    from openmdao.devtools.memory import mem_usage

    verbose = table_format is not None
    results = []
    for method in methods:
        mem_start = mem_usage()
        p = Problem()
        comp = p.model.add_subsystem('comp', comp_class(**kwargs))
        comp.options['derivs_method'] = method
        if method in ('cs', 'fd'):
            comp._has_approx = True
            comp._get_approx_scheme(method)

        if initial_vals:
            for name, val in initial_vals.items():
                p.model.set_val('comp.' + name, val)

        p.setup(mode=mode, force_alloc_complex='cs' in methods)
        p.run_model()

        model_mem = mem_usage

        if verbose:
            print(f"\nModel memory usage: {model_mem} MB")
            print(f"\nTiming {repeats} compute calls for {comp_class.__name__} using "
                  f"{method} method.")
        start = time.perf_counter()
        for n in range(repeats):
            comp.compute(comp._inputs, comp._outputs)
            if verbose:
                print('.', end='', flush=True)
        results.append([method, 'compute', n, time.perf_counter() - start, None])

        diff_mem = mem_usage() - mem_start
        results[-1][-1] = diff_mem

        if verbose:
            print(f"\n\nTiming {repeats} compute_partials calls for {comp_class.__name__} using "
                  f"{method} method.")
        start = time.perf_counter()
        for n in range(repeats):
            p.model._linearize(None)
            if verbose:
                print('.', end='', flush=True)
        results.append([method, 'compute_partials', n, time.perf_counter() - start, None])

        diff_mem = mem_usage() - model_mem
        results[-1][-1] = diff_mem

        del p

    if verbose:
        print('\n')
        headers = ['Method', 'Function', 'Iterations', 'Time (s)', 'Memory (MB)']
        generate_table(results, tablefmt=table_format, headers=headers).display()

    return results


def jax_deriv_shape(derivs):
    """
    Get the shape of the derivatives from a jax derivative calculation.

    Parameters
    ----------
    derivs : tuple
        The tuple of derivatives.

    Returns
    -------
    list
        The shape of the derivatives.
    """
    dims = []
    if isinstance(derivs, jnp.ndarray):
        dims.append(derivs.shape)
    else:   # tuple
        for d in derivs:
            if isinstance(d, jnp.ndarray):
                dims.append(d.shape)
            else:
                dims.append(jax_deriv_shape(d))
    return dims


if jax is None or bool(os.environ.get('JAX_DISABLE_JIT', '')):
    def DelayedJit(func):
        """
        Do nothing.

        Parameters
        ----------
        func : Callable
            The function or method to be wrapped.

        Returns
        -------
        Callable
            The original function or method.

        """
        return func

    def _jax_update_class_attrs(name, bases, attrs):
        pass

    def _jax_register_class(cls, name, bases, attrs):
        pass

else:

    def _jax_update_class_attrs(name, bases, attrs):
        """
        Add _tree_flatten and _tree_unflatten methods if they're missing.

        If a class has 'self static' attributes, it must define the 'get_self_statics' method,
        which returns a tuple of static attributes of the class.

        A 'self static' attribute is one that is accessed on 'self' within the jitted method
        but is not passed in as an argument to the method.


        Parameters
        ----------
        name : str
            The name of the class.
        bases : tuple
            The base classes of the class.
        attrs : dict
            The attributes of the class.
        """
        attrs['_tree_flatten'] = lambda self: ((), {'_self_': self,
                                                    '_statics_': self.get_self_statics()})
        attrs['_tree_unflatten'] = \
            staticmethod(lambda aux_data, children: aux_data['_self_'])

    def _jax_register_class(cls, name, bases, attrs):
        """
        Register a class with jax so that it can be used with jax.jit.

        Parameters
        ----------
        cls : class
            The class to be registered.
        name : str
            The name of the class.
        bases : tuple
            The base classes of the class.
        attrs : dict
            The attributes of the class.
        """
        # register with jax so we can flatten/unflatten self
        tree_util.register_pytree_node(cls, attrs['_tree_flatten'], attrs['_tree_unflatten'])

    class DelayedJit:
        """
        Wrap the method of a class, providing a delayed jit capability.

        This decorator is used to delay the jit compilation of a method until the first time it is
        called. This allows the method to be compiled with the correct static arguments, which can
        be determined automatically on the first call. This is useful for methods of a class
        that are jit compiled and that reference static attributes of the class.

        Parameters
        ----------
        func : Callable
            The function or method to be wrapped.

        Attributes
        ----------
        _func : Callable
            The function or method to be wrapped.
        _jfunc : Callable
            The jitted function.
        """

        def __init__(self, func):
            """
            Initialize the DelayedJit object.

            Parameters
            ----------
            func : Callable
                The function or method to be wrapped.
            """
            self._func = func
            self._jfunc = None  # jitted function

        def __call__(self, *args, **kwargs):
            """
            Jit the function on the first call after we know which args are static.

            Parameters
            ----------
            args : list
                Positional arguments.
            kwargs : dict
                Keyword arguments.

            Returns
            -------
            object
                The result of the function call.
            """
            if self._jfunc is None:
                params = list(inspect.signature(self._func).parameters)
                # we don't want to treat 'self' as static because we want _tree_(flatten/unflatten)
                # to be called on it so 'get_self_statics' will be called.
                offset = 1 if params and params[0] == 'self' else 0
                # static args are those that are not jax or numpy arrays
                static_argnums = [i for (i, arg) in enumerate(args)
                                  if i >= offset and not isinstance(arg, (Array, ndarray))]
                static_argnames = [n for (n, v) in kwargs.items()
                                   if not isinstance(v, (Array, ndarray))]
                self._jfunc = jit(self._func, static_argnums=static_argnums,
                                  static_argnames=static_argnames)
            return self._jfunc(*args, **kwargs)


# we define compute_partials here instead of making this the base class version as we
# did with compute, because the existence of a compute_partials method that is not the
# base class method is used to determine if a given component computes its own partials.
def compute_partials(self, inputs, partials, discrete_inputs=None):
    """
    Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

    Parameters
    ----------
    self : Component
        The component instance.
    inputs : Vector
        Unscaled, dimensional input variables read via inputs[key].
    partials : Jacobian
        Sub-jac components written to partials[output_name, input_name]..
    discrete_inputs : dict or None
        If not None, dict containing discrete input values.
    """
    deriv_vals = self._get_jac_func()(*self._get_compute_primal_invals(inputs, discrete_inputs))
    nested_tup = isinstance(deriv_vals, tuple) and len(deriv_vals) > 0 and \
        isinstance(deriv_vals[0], tuple)

    nof = len(self._var_rel_names['output'])
    ofidx = len(self._discrete_outputs) - 1
    for ofname in self._var_rel_names['output']:
        ofidx += 1
        ofmeta = self._var_rel2meta[ofname]
        for wrtidx, wrtname in enumerate(self._var_rel_names['input']):
            key = (ofname, wrtname)
            if key not in partials:
                # FIXME: this means that we computed a derivative that we didn't need
                continue

            wrtmeta = self._var_rel2meta[wrtname]
            dvals = deriv_vals
            # if there's only one 'of' value, we only take the indexed value if the
            # return value of compute_primal is single entry tuple. If a single array or
            # scalar is returned, we don't apply the 'of' index.
            if nof > 1 or nested_tup:
                dvals = dvals[ofidx]

            dvals = dvals[wrtidx].reshape(ofmeta['size'], wrtmeta['size'])

            sjmeta = partials.get_metadata(key)
            rows = sjmeta['rows']
            if rows is None:
                partials[ofname, wrtname] = dvals
            else:
                partials[ofname, wrtname] = dvals[rows, sjmeta['cols']]


def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
    r"""
    Compute jac-vector product. The model is assumed to be in an unscaled state.

    If mode is:
        'fwd': d_inputs \|-> d_outputs

        'rev': d_outputs \|-> d_inputs

    Parameters
    ----------
    self : Component
        The component instance.
    inputs : Vector
        Unscaled, dimensional input variables read via inputs[key].
    d_inputs : Vector
        See inputs; product must be computed only if var_name in d_inputs.
    d_outputs : Vector
        See outputs; product must be computed only if var_name in d_outputs.
    mode : str
        Either 'fwd' or 'rev'.
    discrete_inputs : dict or None
        If not None, dict containing discrete input values.
    """
    if mode == 'fwd':
        _, deriv_vals = jax.jvp(self.compute_primal,
                                primals=tuple(self._get_compute_primal_invals(inputs,
                                                                              discrete_inputs)),
                                tangents=tuple(d_inputs.values()))

        d_outputs.set_vals(deriv_vals)
    else:
        # TODO: cache the vjp_fun at each NL point since the inputs won't change during the
        # computation of the jacobian rows.
        _, vjp_fun = jax.vjp(self.compute_primal,
                             *self._get_compute_primal_invals(inputs, discrete_inputs))

        d_inputs.set_vals(vjp_fun(tuple(d_outputs.values())))


if __name__ == '__main__':
    # import openmdao.api as om

    # def func(x, y):  # noqa: D103
    #     z = jnp.sin(x) * y
    #     q = x * 1.5
    #     zz = q + x * 1.5
    #     return z, zz

    # shape = (3, 2)
    # x = jnp.ones(shape)
    # y = jnp.ones(shape) * 2.0
    # jaxpr = jax.make_jaxpr(func)(x, y)

    # dump_jaxpr(jaxpr)

    # print(list(get_partials_deps(func, ('z', 'zz'), x, y)))

    # jac_func = jax.jacfwd(func)

    # jaxpr = jax.make_jaxpr(jac_func)(x, y)
    # dump_jaxpr(jaxpr)

    # print(list(get_partials_deps(jac_func, ('z', 'zz'), x, y)))

    # p = om.Problem()
    # comp = p.model.add_subsystem('comp', om.ExecComp('y = 2.0*x', x=np.ones(3), y=np.ones(3)))
    # comp.derivs_method='jax'
    # p.setup()
    # p.run_model()

    # print(p.compute_totals(of=['comp.y'], wrt=['comp.x']))

    from openmdao.components.mux_comp import MuxComp

    c = MuxComp()

    sf = SelfAttrFinder(c.compute)
