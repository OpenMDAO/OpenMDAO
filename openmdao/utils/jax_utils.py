"""
Utilities for the use of jax in combination with OpenMDAO.
"""
import ast
import textwrap
import inspect


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
    import jax.numpy as jnp
    # Importing Jax functions useful for tracing/interpreting.
    from jax import core
    from jax import lax
    from jax._src.util import safe_map
    from jax._src.core import eval_jaxpr
    from jax import jit as jjit
except ImportError:
    jax = None
    jjit = jit_stub

import numpy as np
from functools import wraps, partial
import openmdao.func_api as omf




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


class Compute2Jax(ast.NodeTransformer):
    """
    An ast.NodeTransformer that transforms a compute function definition to jax compatible form.

    """
    def __init__(self, func):
        self._args = list(inspect.signature(func).parameters)
        self._dict_keys = {arg: [] for arg in self._args}
        node = self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec'))
        newast = ast.fix_missing_locations(node)
        # print(ast.unparse(newast))
        code = compile(newast, '<ast>', 'exec')
        namespace = func.__globals__.copy()  # don't clutter up the original namespace
        exec(code, namespace)
        self._transformed = jjit(namespace[func.__name__], static_argnums=(0,))

    def _get_input_names(self):
        return self._dict_keys[self._args[1]]

    def _get_input_names_src(self):
        return ', '.join([f"'{n}'" for n in self._get_input_names()])

    def _get_output_names(self):
        return self._dict_keys[self._args[2]]

    def _get_output_names_src(self):
        return ', '.join([f"'{n}'" for n in self._get_output_names()])

    # def dump(self):
    #     print(ast.unparse(self._transformed))
    #     print("dict keys:")
    #     import pprint
    #     pprint.pprint(self._dict_keys)

    def get_new_args(self):
        new_args = [ast.arg('self', annotation=None)]
        for arg_name in self._get_input_names():
            new_args.append(ast.arg(arg=arg_name, annotation=None))
        return ast.arguments(args=new_args, posonlyargs=[], vararg=None, kwonlyargs=[],
                             kw_defaults=[], kwarg=None, defaults=[])

    def _make_return(self, names):
        val = ast.Tuple([ast.Name(id=n, ctx=ast.Load()) for n in names], ctx=ast.Load())
        return ast.Return(val)

    def visit_FunctionDef(self, node):
        newbody = [self.visit(statement) for statement in node.body]
        # add a return statement for the outputs
        newbody.append(self._make_return(self._get_output_names()))

        newargs = self.get_new_args()
        return ast.FunctionDef(node.name, newargs, newbody, node.decorator_list, node.returns,
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
        if (isinstance(node.value, ast.Name) and node.value.id in self._args and
                isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            self._dict_keys[node.value.id].append(node.slice.value)
            return ast.copy_location(ast.Name(id=node.slice.value, ctx=node.ctx), node)
        else:
            return self.generic_visit(node)


def _get_new_compute(comp):
    # TODO: ordering here will be an issue...
    src = f"""
def compute(self, inputs, outputs):
    # print('ins:', list(inputs.values()))
    # print('outs:', self.compute_primal(*inputs.values()))
    outputs.set_vals(*self.compute_primal(*inputs.values()))
"""
    node = ast.parse(src, mode='exec')
    code = compile(node, '<string>', 'exec')
    # print("COMPUTE:")
    # print(ast.dump(node))
    namespace = {}
    exec(code, namespace)
    return namespace['compute']


def _get_compute_partials(compute_info):
    src = f"""
def compute_partials(self, inputs, partials):
    deriv_func = self._get_deriv_func_()
    deriv_vals = deriv_func(*inputs.values())[0]
    ideriv = 0
    for ofname in [{compute_info._get_output_names_src()}]:
        for wrtname in [{compute_info._get_input_names_src()}]:
            print("SETTING PARTIALS:", self._mode, ofname, wrtname)
            print(deriv_vals[0][ideriv])
            partials[ofname, wrtname] = deriv_vals[ideriv]
            ideriv += 1
"""
    # print("SRC:")
    # print(src)
    node = ast.parse(src, mode='exec')
    code = compile(node, '<string>', 'exec')
    namespace = {}
    exec(code, namespace)
    return namespace['compute_partials']


def _get_deriv_func(compute_info):
    argnums = list(range(len(compute_info._get_input_names())))
    src = f"""
def _get_deriv_func_(self):
    try:
        cache = self._deriv_func_cache_
    except AttributeError:
        cache = self._deriv_func_cache_ = {{}}
    mode = self._mode

    if mode not in cache:
        if mode == 'fwd':
            cache[mode] = jax.jacfwd(self.compute_primal, argnums={argnums})
        else:  # mode == 'rev'")
            cache[mode] = jax.jacrev(self.compute_primal, argnums={argnums})
    return cache[mode]
"""
    node = ast.parse(src, mode='exec')
    code = compile(node, '<string>', 'exec')
    namespace = {'jax': jax}
    exec(code, namespace)
    return namespace['_get_deriv_func_']


# TODO: make this a class decorator that can take args so we can tell it which jax deriv
# method(s) to use.
def jaxify_component(comp_class):
    """
    Jaxify a normal OpenMDAO component class.

    Parameters
    ----------
    comp_class : class
        The class to be decorated.

    Returns
    -------
    class
        The decorated class.
    """
    compute_info = Compute2Jax(comp_class.compute)
    setattr(comp_class, 'compute_primal', compute_info._transformed)
    setattr(comp_class, 'compute', _get_new_compute(comp_class))
    setattr(comp_class, 'compute_partials', _get_compute_partials(compute_info))
    setattr(comp_class, '_get_deriv_func_', _get_deriv_func(compute_info))

    return comp_class


if __name__ == '__main__':
    import openmdao.api as om

    # @ jax.tree_util.register_pytree_node_class
    @jaxify_component
    class MyComp(om.ExplicitComponent):
        def setup(self):
            self.add_input('x', shape=(3, 3))
            self.add_input('y', shape=(3, 4))
            self.add_output('z', shape=(3, 4))

            self.declare_partials(of='z', wrt=['x', 'y'])

        def compute(self, inputs, outputs):
            outputs['z'] = jnp.dot(inputs['x'], inputs['y'])

    p = om.Problem()
    comp = p.model.add_subsystem('comp', MyComp())
    p.setup(mode='rev')

    x = np.arange(1,10).reshape((3,3))
    y = np.arange(1,13).reshape((3,4))
    p.set_val('comp.x', x)
    p.set_val('comp.y', y)
    p.final_setup()
    p.run_model()
    p.check_totals(of=['comp.z'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
    p.check_partials(show_only_incorrect=True)

    # inputs = {'x': jnp.ones((5, 4))*5., 'y': jnp.ones((4, 7))*3.}
    # outputs = {'z': jnp.zeros((5, 7))}
    # closed_jaxpr = jax.make_jaxpr(f)(inputs, outputs)
    # examine_jaxpr(closed_jaxpr)
    # print(eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, inputs['x'], inputs['y'], outputs['z']))
    # print("inputs:", list(comp._inputs.items()))
    # comp.compute(comp._inputs, comp._outputs)
    print(np.dot(x, y))
    print(list(comp._outputs.items()))
