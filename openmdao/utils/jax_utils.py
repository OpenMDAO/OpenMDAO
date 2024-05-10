"""
Utilities for the use of jax in combination with OpenMDAO.
"""
import ast
import textwrap
import inspect
from itertools import chain

import networkx as nx


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
    from jax import jit as jjit
except ImportError:
    jax = None
    jjit = jit_stub

import numpy as np


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


# def eval_jaxpr(jaxpr, consts, *args):
#   # Mapping from variable -> value
#   env = {}

#   def read(var):
#     # Literals are values baked into the Jaxpr
#     if type(var) is core.Literal:
#       return var.val
#     return env[var]

#   def write(var, val):
#     env[var] = val

#   # Bind args and consts to environment
#   safe_map(write, jaxpr.invars, args)
#   safe_map(write, jaxpr.constvars, consts)

#   # Loop through equations and evaluate primitives using `bind`
#   for eqn in jaxpr.eqns:
#     # Read inputs to equation from environment
#     invals = safe_map(read, eqn.invars)
#     # `bind` is how a primitive is called
#     outvals = eqn.primitive.bind(*invals, **eqn.params)
#     # Primitives may return multiple outputs or not
#     if not eqn.primitive.multiple_results:
#       outvals = [outvals]
#     # Write the results of the primitive into the environment
#     safe_map(write, eqn.outvars, outvals)
#   # Read the final result of the Jaxpr from the environment
#   return safe_map(read, jaxpr.outvars)

def examine_jaxpr(closed_jaxpr):
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

    # show the function graph visually
    # from openmdao.visualization.graph_viewer import write_graph, _to_pydot_graph
    # G = _to_pydot_graph(graph)
    # write_graph(G)

    return graph


def get_partials_deps(func, outputs, *args, **kwargs):
    graph = get_func_graph(func, *args, **kwargs)
    inputs = list(inspect.signature(func).parameters)
    n2index = {n: i for i, n in enumerate(chain(inputs, outputs))}
    for inp in inputs:
        for out in outputs:
            # TODO: not sure how efficient this is...
            if nx.has_path(graph, n2index[inp], n2index[out]):
                yield (out, inp)


# TODO: discrete vars not supported yet
class Compute2Jax(ast.NodeTransformer):
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

    Attributes
    ----------
    _args : list
        The argument names of the original compute function.
    _dict_keys : dict
        A dictionary that maps each argument name to a list of subscript values.
    _transformed : function
        The transformed compute function.
    """
    # these ops require static objects so their args should not be traced.  Traced array ops should
    # use jnp and static ones should use np.
    _static_ops = {'reshape'}
    _np_names = {'np', 'numpy'}

    def __init__(self, comp, use_jit=True, verbose=False):
        func = comp.compute
        self._namespace = comp.compute.__globals__.copy()
        if 'jnp' not in self._namespace:
            self._namespace['jnp'] = jnp
        self._orig_args = list(inspect.signature(func).parameters)

        # ensure that ordering of args and returns exactly matches the order of the inputs and
        # outputs vectors.
        self._args = [n for n in chain(comp._discrete_inputs, comp._var_rel_names['input'])]
        self._returns = [n for n in chain(comp._discrete_outputs, comp._var_rel_names['output'])]
        self._discrete_outs = list(comp._discrete_outputs)

        self._dict_keys = {arg: set() for arg in self._orig_args}
        node = self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec'))
        newast = ast.fix_missing_locations(node)

        # check that inputs/outputs referenced in the function match those declared in the component
        if set(self._dict_keys['inputs']) != set(comp._var_rel_names['input']):
            raise RuntimeError(f"{comp.msginfo}: The inputs referenced in the compute method "
                               f"{sorted(self._dict_keys['inputs'])} do not match the inputs "
                               f"declared in the component {sorted(comp._var_rel_names['input'])}.")
        if set(self._dict_keys['outputs']) != set(comp._var_rel_names['output']):
            raise RuntimeError(f"{comp.msginfo}: The outputs referenced in the compute method "
                               f"{sorted(self._dict_keys['outputs'])} do not match the outputs "
                               f"declared in the component {sorted(comp._var_rel_names['output'])}."
                               )

        if verbose:
            print("Converted function:")
            print(ast.unparse(newast))

        code = compile(newast, '<ast>', 'exec')
        exec(code, self._namespace)
        self._transformed = self._namespace['compute_primal']
        if use_jit:
            statics = tuple(range(len(comp._var_discrete['input']) + 1))
            self._transformed = jjit(self._transformed, static_argnums=statics)

    def get_new_args(self):
        new_args = [ast.arg('self', annotation=None)]
        for arg_name in self._args:
            new_args.append(ast.arg(arg=arg_name, annotation=None))
        return ast.arguments(args=new_args, posonlyargs=[], vararg=None, kwonlyargs=[],
                             kw_defaults=[], kwarg=None, defaults=[])

    def get_pre_body(self):
        if not self._discrete_outs:
            return []

        # add a statement to pull individual values out of the discrete outputs
        elts = [ast.Name(id=name, ctx=ast.Store()) for name in self._discrete_outs]
        return [ast.Assign(targets=[ast.Tuple(elts=elts, ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(
                               value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                   attr='_discrete_outputs', ctx=ast.Load()),
                                                   attr='values', ctx=ast.Load()),
                                                   args=[], keywords=[]))]

    def get_post_body(self):
        if not self._discrete_outs:
            return []

        # add a statement to set the values of self._discrete outputs
        args = [ast.Name(id=name, ctx=ast.Load()) for name in self._discrete_outs]
        return [ast.Expr(value=ast.Call(func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                attr='_discrete_outputs', ctx=ast.Load()),
                                attr='set_vals', ctx=ast.Load()), args=args, keywords=[]))]

    def _make_return(self):
        val = ast.Tuple([ast.Name(id=n, ctx=ast.Load()) for n in self._returns], ctx=ast.Load())
        return ast.Return(val)

    def visit_FunctionDef(self, node):
        newbody = self.get_pre_body()
        newbody.extend([self.visit(statement) for statement in node.body])
        newbody.extend(self.get_post_body())
        # add a return statement for the outputs
        newbody.append(self._make_return())

        newargs = self.get_new_args()
        return ast.FunctionDef('compute_primal', newargs, newbody, node.decorator_list, node.returns,
                               node.type_comment)

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
        # if we encounter a subscript of the 'inputs' arg (it may not be named that) then replace
        # inputs['name'] with name.
        if (isinstance(node.value, ast.Name) and node.value.id in self._orig_args and
                isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            self._dict_keys[node.value.id].add(node.slice.value)
            return ast.copy_location(ast.Name(id=node.slice.value, ctx=node.ctx), node)

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
                if 'jnp' in self._namespace:
                    return ast.copy_location(ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()),
                                                           attr=node.attr, ctx=node.ctx), node)

        return self.generic_visit(node)


if __name__ == '__main__':
    def func(x, y):
        z = jnp.sin(x) * y
        q = x * 1.5
        zz = q + x * 1.5
        return z, zz

    shape = (3, 2)
    x =  jnp.ones(shape)
    y =  jnp.ones(shape) * 2.0
    jaxpr = jax.make_jaxpr(func)(x, y)

    examine_jaxpr(jaxpr)

    print(list(get_partials_deps(func, ('z', 'zz'), x, y)))

    jac_func = jax.jacfwd(func)

    jaxpr = jax.make_jaxpr(jac_func)(x, y)
    examine_jaxpr(jaxpr)

    print(list(get_partials_deps(jac_func, ('z', 'zz'), x, y)))
