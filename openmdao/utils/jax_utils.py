"""
Utilities for the use of jax in combination with OpenMDAO.
"""
import re
import ast
import textwrap
import inspect
import weakref
from itertools import chain

import networkx as nx
import numpy as np

from openmdao.visualization.tables.table_builder import generate_table


try:
    import jax
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
    import jax.numpy as jnp
    from jax import jit
except ImportError:
    jax = None
    jnp = np

    def jit(f, *args, **kwargs):
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
            if type(inp) == jax._src.core.Var:
                inp = str(inp)
                if inp not in n2index:
                    n2index[inp] = i
                    graph.add_node(i, label=inp)
                    i += 1
                for out in eqn.outvars:
                    if type(out) == jax._src.core.Var:
                        out = str(out)
                        if out not in n2index:
                            n2index[out] = i
                            graph.add_node(i, label=out)
                            i += 1
                        graph.add_edge(n2index[inp], n2index[out])

    return graph


def get_partials_deps(func, outputs, *args, **kwargs):
    """
    Generate a list of tuples of the form (output, input) for the given function.

    Only tuples where the output depends on the input are yielded. This can be used to
    determine which partials need to be declared.

    Parameters
    ----------
    func : Callable
        The function to be analyzed.
    outputs : list
        The list of output variable names.
    *args : list
        Positional arguments.
    **kwargs : dict
        Keyword arguments.

    Yields
    ------
    tuple
        A tuple of the form (output, input).
    """
    graph = get_func_graph(func, *args, **kwargs)

    # show the function graph visually
    # from openmdao.visualization.graph_viewer import write_graph, _to_pydot_graph
    # G = _to_pydot_graph(graph)
    # write_graph(G)

    inputs = list(inspect.signature(func).parameters)
    n2index = {n: i for i, n in enumerate(chain(inputs, outputs))}
    for inp in inputs:
        for out in outputs:
            # TODO: not sure how efficient this is...
            if nx.has_path(graph, n2index[inp], n2index[out]):
                yield (out, inp)


class ExplicitCompJaxify(ast.NodeTransformer):
    """
    An ast.NodeTransformer that transforms a compute function definition to jax compatible form.

    So compute(self, inputs, outputs) becomes f(self, *args) where args are the input values
    in the order they are stored in inputs.  The new function will return a tuple of the
    output values in the order they are stored in outputs.

    Parameters
    ----------
    comp : ExplicitComponent
        The Component whose compute function is to be transformed. This NodeTransformer may only
        be used after the Component has had its _setup_var_data method called, because that
        determines the ordering of the inputs and outputs.
    use_jit : bool
        If True, the transformed function will be jitted.
    verbose : bool
        If True, the transformed function will be printed to stdout.

    Attributes
    ----------
    _comp : weakref.ref
        A weak reference to the Component whose compute function is being transformed.
    _compute_primal_args : list
        The argument names of the compute_primal function.
    _compute_primal_returns : list
        The names of the outputs from the new function.
    compute_primal : function
        The compute_primal function created from the original compute function.
    _compute_args : list
        The original argument names of the compute function.
    _new_ast : ast node
        The new ast node created from the original compute function.
    """

    # these ops require static objects so their args should not be traced.  Traced array ops should
    # use jnp and static ones should use np.
    _static_ops = {'reshape'}
    _np_names = {'np', 'numpy'}

    def __init__(self, comp, use_jit=True, verbose=False):  # noqa: D107
        self._comp = weakref.ref(comp)
        func = comp.compute
        if 'jnp' not in comp.compute.__globals__:
            comp.compute.__globals__['jnp'] = jnp
        namespace = comp.compute.__globals__.copy()

        self._compute_args = list(inspect.signature(func).parameters)

        # ensure that ordering of args and returns exactly matches the order of the inputs and
        # outputs vectors.
        self._compute_primal_args = \
            [n for n in chain(comp._discrete_inputs, comp._var_rel_names['input'])]
        self._compute_primal_returns = \
            [n for n in chain(comp._discrete_outputs, comp._var_rel_names['output'])]

        node = self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec'))
        self._new_ast = ast.fix_missing_locations(node)

        code = compile(self._new_ast, '<ast>', 'exec')
        exec(code, namespace)
        self.compute_primal = namespace['compute_primal']

        if verbose:
            print(f"\n{self.get_new_source()}\n")

        if use_jit:
            static_argnums = tuple(range(len(comp._var_discrete['input']) + 1))
            self.compute_primal = \
                    jit(self.compute_primal, static_argnums=static_argnums)

    def get_new_source(self):
        """
        Return the source code of the transformed function.

        Returns
        -------
        str
            The source code of the transformed function.
        """
        return ast.unparse(self._new_ast)

    def _get_arg_values(self):
        discrete_inputs = self._comp()._discrete_inputs
        yield from discrete_inputs.values()
        if self._comp()._inputs is None:
            for name, meta in self._comp()._var_rel2meta['input'].items():
                if name not in discrete_inputs:
                    yield meta['value']
        else:
            yield from self._comp()._inputs.values()

    def compute_dependency_iter(self):
        """
        Get (output, input) pairs where the output depends on the input for the compute function.

        Yields
        ------
        tuple
            A tuple of the form (output, input), where output depends on the input.
        """
        yield from get_partials_deps(self.compute_primal, self._compute_primal_returns,
                                     *self._get_arg_values())

    def _get_new_args(self):
        new_args = [ast.arg('self', annotation=None)]
        for arg_name in self._compute_primal_args:
            new_args.append(ast.arg(arg=arg_name, annotation=None))
        return ast.arguments(args=new_args, posonlyargs=[], vararg=None, kwonlyargs=[],
                             kw_defaults=[], kwarg=None, defaults=[])

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

    def visit_FunctionDef(self, node):
        """
        Transform the compute function definition.

        The function will be transformed from compute(self, inputs, outputs) to
        compute_primal(self, *args) where args are the input values in the order they are stored in
        inputs.  All subscript accesses into the input args will be replaced with the name of the
        key being accessed, e.g., inputs['foo'] becomes foo. The new function will return a tuple
        of the output values in the order they are stored in outputs.  If compute has the
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
        # if we encounter a subscript of any of the input args, then replace arg['name'] with name.
        if (isinstance(node.value, ast.Name) and node.value.id in self._compute_args and
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


_replace = set((':', '(', ')', '[', ']', '{', '}', ' ', '-',
                '+', '*', '/', '^', '%', '!', '<', '>', '='))


def _fixname(name):
    """
    Ensure that the given name is a valid Python variable name.

    Parameters
    ----------
    name : str
        The name to be fixed.

    Returns
    -------
    str
        The fixed name.
    """
    intr = _replace.intersection(name)
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
        model_diff_mem = model_mem - mem_start

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


if __name__ == '__main__':
    import openmdao.api as om

    def func(x, y):  # noqa: D103
        z = jnp.sin(x) * y
        q = x * 1.5
        zz = q + x * 1.5
        return z, zz

    shape = (3, 2)
    x = jnp.ones(shape)
    y = jnp.ones(shape) * 2.0
    jaxpr = jax.make_jaxpr(func)(x, y)

    dump_jaxpr(jaxpr)

    print(list(get_partials_deps(func, ('z', 'zz'), x, y)))

    jac_func = jax.jacfwd(func)

    jaxpr = jax.make_jaxpr(jac_func)(x, y)
    dump_jaxpr(jaxpr)

    print(list(get_partials_deps(jac_func, ('z', 'zz'), x, y)))

    p = om.Problem()
    comp = p.model.add_subsystem('comp', om.ExecComp('y = 2.0*x', x=np.ones(3), y=np.ones(3)))
    comp.derivs_method='jax'
    p.setup()
    p.run_model()

    print(p.compute_totals(of=['comp.y'], wrt=['comp.x']))

