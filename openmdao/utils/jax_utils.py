"""
Utilities for the use of jax in combination with OpenMDAO.
"""
import sys
import os
import ast
import textwrap
import inspect
import weakref
from itertools import chain
from collections import defaultdict
import importlib

import numpy as np

from openmdao.visualization.tables.table_builder import generate_table
from openmdao.utils.code_utils import _get_long_name, remove_src_blocks, replace_src_block, \
    get_partials_deps
from openmdao.utils.file_utils import get_module_path, _load_and_exec


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
    from jax import jit, tree_util
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
        try:
            class_src = textwrap.dedent(inspect.getsource(self._comp().__class__))
        except Exception:
            raise RuntimeError(f"Couldn't obtain class source for {self._comp().__class__}.")

        compute_primal_src = textwrap.indent(textwrap.dedent(self.get_compute_primal_src()),
                                             ' ' * 4)

        class_src = replace_src_block(class_src, self._funcname, compute_primal_src,
                                      block_start_tok='def')
        class_src = remove_src_blocks(class_src, self._get_del_methods(), block_start_tok='def')

        return class_src.rstrip()

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
        val = ast.Tuple([ast.Name(id=n, ctx=ast.Load())
                         for n in self._get_compute_primal_returns()], ctx=ast.Load())
        return ast.Return(val)

    def _get_new_args(self):
        new_args = [ast.arg('self', annotation=None)]
        for arg_name in self._get_compute_primal_args():
            new_args.append(ast.arg(arg=arg_name, annotation=None))
        return ast.arguments(args=new_args, posonlyargs=[], vararg=None, kwonlyargs=[],
                             kw_defaults=[], kwarg=None, defaults=[])

    def visit_FunctionDef(self, node):
        """
        Transform the compute function definition.

        The function will be transformed from compute(self, inputs, outputs, ...) or
        apply_nonlinear(self, ...) to compute_primal(self, arg1, arg2, ...) where args are the
        input values in the order they are stored in inputs.  All subscript accesses into the input
        args will be replaced with the name of the key being accessed, e.g., inputs['foo'] becomes
        foo. The new function will return a tuple of the output values in the order they are stored
        in outputs.  If compute has the additional args discrete_inputs and discrete_outputs, they
        will be handled similarly.

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
        for statement in node.body:
            newnode = self.visit(statement)
            if newnode is not None:
                newbody.append(newnode)
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

    def visit_Assign(self, node):
        """
        Translate an Assign node into an Assign node with the subscript replaced with the name.

        Parameters
        ----------
        node : ast.Assign
            The Assign node being visited.

        Returns
        -------
        ast.Any
            The transformed node.
        """
        if len(node.targets) == 1:
            nodeval = self.visit(node.value)
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(nodeval, ast.Name):
                if tgt.id == nodeval.id:
                    return None  # get rid of any 'x = x' assignments after conversion

        return self.generic_visit(node)


class ExplicitCompJaxify(CompJaxifyBase):
    """
    An ast.NodeTransformer that transforms a compute function definition to jax compatible form.

    So compute(self, inputs, outputs) becomes compute_primal(self, arg1, arg2, ...) where args are
    the input values in the order they are stored in inputs.  The new function will return a tuple
    of the output values in the order they are stored in outputs.

    If the component has discrete inputs, they will be passed individually into compute_primal
    *after* the continuous inputs.  If the component has discrete outputs, they will be assigned
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

    def _get_compute_primal_args(self):
        # ensure that ordering of args and returns exactly matches the order of the inputs and
        # outputs vectors.
        return chain(self._comp()._var_rel_names['input'], self._comp()._discrete_inputs)

    def _get_compute_primal_returns(self):
        return chain(self._comp()._var_rel_names['output'], self._comp()._discrete_outputs)

    def _get_del_methods(self):
        return ['compute', 'compute_partials', 'compute_jacvec_product']


class ImplicitCompJaxify(CompJaxifyBase):
    """
    A NodeTransformer that transforms an apply_nonlinear function definition to jax compatible form.

    So apply_nonlinear(self, inputs, outputs, residuals) becomes
    compute_primal(self, arg1, arg2, ...) where args are
    the input and output values in the order they are stored in their respective Vectors.
    The new function will return a tuple of the residual values in the order they are stored in
    the residuals Vector.

    If the component has discrete inputs, they will be passed individually into compute_primal
    *after* the continuous inputs.  If the component has discrete outputs, they will be assigned
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

    def _get_compute_primal_args(self):
        # ensure that ordering of args and returns exactly matches the order of the inputs,
        # outputs, and residuals vectors.
        return chain(self._comp()._var_rel_names['input'], self._comp()._var_rel_names['output'],
                     self._comp()._discrete_inputs)

    def _get_compute_primal_returns(self):
        return chain(self._comp()._var_rel_names['output'], self._comp()._discrete_outputs)

    def _get_del_methods(self):
        return ['apply_nonlinear', 'linearize', 'apply_linear']


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


class ReturnChecker(ast.NodeVisitor):
    """
    An ast.NodeVisitor that determines if a method returns a tuple or not.

    Parameters
    ----------
    method : method
        The method to be analyzed.

    Attributes
    ----------
    _returns : list
        The list of boolean values indicating whether or not the method returns a tuple. One
        entry for each return statement in the method.
    """

    def __init__(self, method):  # noqa
        self._returns = []
        self.visit(ast.parse(textwrap.dedent(inspect.getsource(method)), mode='exec'))

    def returns_tuple(self):
        """
        Return whether or not the method returns a tuple.

        Returns
        -------
        bool
            True if the method returns a tuple, False otherwise.
        """
        if self._returns:
            ret = self._returns[0]
            for r in self._returns[1:]:
                if r != ret:
                    raise RuntimeError("ReturnChecker can't handle a method with multiple return "
                                       "statements that return different types.")
            return ret
        return False

    def visit_Return(self, node):
        """
        Visit a Return node.

        Parameters
        ----------
        node : ASTnode
            The return node being visited.
        """
        self._returns.append(isinstance(node.value, ast.Tuple))


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
    def _jax_register_pytree_class(cls):
        pass

else:

    _registered_classes = set()

    def _jax_register_pytree_class(cls):
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
        global _registered_classes
        if cls not in _registered_classes:
            # register with jax so we can flatten/unflatten self
            tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)
            _registered_classes.add(cls)


# we define compute_partials here instead of making this the base class version as we
# did with compute, because the existence of a compute_partials method that is not the
# base class method is used to determine if a given component computes its own partials.
def compute_partials(inst, inputs, partials, discrete_inputs=None):
    """
    Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

    Parameters
    ----------
    inst : ImplicitComponent
        The component instance.
    inputs : Vector
        Unscaled, dimensional input variables read via inputs[key].
    partials : Jacobian
        Sub-jac components written to partials[output_name, input_name]..
    discrete_inputs : dict or None
        If not None, dict containing discrete input values.
    """
    deriv_vals = inst._get_jac_func()(*inst._get_compute_primal_invals(inputs,
                                                                       inst._discrete_inputs))
    nested_tup = isinstance(deriv_vals, tuple) and len(deriv_vals) > 0 and \
        isinstance(deriv_vals[0], tuple)

    nof = len(inst._var_rel_names['output'])
    for ofidx, ofname in enumerate(inst._var_rel_names['output']):
        ofmeta = inst._var_rel2meta[ofname]
        for wrtidx, wrtname in enumerate(inst._var_rel_names['input']):
            key = (ofname, wrtname)
            if key not in partials:
                # FIXME: this means that we computed a derivative that we didn't need
                continue

            wrtmeta = inst._var_rel2meta[wrtname]
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


def compute_jacvec_product(inst, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
    r"""
    Compute jac-vector product. The model is assumed to be in an unscaled state.

    If mode is:
        'fwd': d_inputs \|-> d_outputs

        'rev': d_outputs \|-> d_inputs

    Parameters
    ----------
    inst : ImplicitComponent
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
        dx = tuple(d_inputs.values())
        full_invals = tuple(inst._get_compute_primal_invals(inputs, discrete_inputs))
        x = full_invals[:len(dx)]
        other = full_invals[len(dx):]
        _, deriv_vals = jax.jvp(lambda *args: inst.compute_primal(*args, *other),
                                primals=x, tangents=dx)
        d_outputs.set_vals(deriv_vals)
    else:
        inhash = ((inputs.get_hash(),) + tuple(inst._discrete_inputs.values()) +
                  inst.get_self_statics())
        if inhash != inst._vjp_hash:
            dx = tuple(d_inputs.values())
            full_invals = tuple(inst._get_compute_primal_invals(inputs, discrete_inputs))
            x = full_invals[:len(dx)]
            other = full_invals[len(dx):]
            # recompute vjp function if inputs have changed
            _, inst._vjp_fun = jax.vjp(lambda *args: inst.compute_primal(*args, *other), *x)
            inst._vjp_hash = inhash

        if inst._compute_primal_returns_tuple:
            deriv_vals = inst._vjp_fun(tuple(d_outputs.values()) +
                                       tuple(inst._discrete_outputs.values()))
        else:
            deriv_vals = inst._vjp_fun(tuple(d_outputs.values())[0])

        d_inputs.set_vals(deriv_vals)


# we define linearize here instead of making this the base class version as we
# did with apply_nonlinear, because the existence of a linearize method that is not the
# base class method is used to determine if a given component computes its own partials.
def linearize(inst, inputs, outputs, partials, discrete_inputs=None, discrete_outputs=None):
    """
    Compute sub-jacobian parts and any applicable matrix factorizations.

    The model is assumed to be in an unscaled state.

    Parameters
    ----------
    inst : ImplicitComponent
        The component instance.
    inputs : Vector
        Unscaled, dimensional input variables read via inputs[key].
    outputs : Vector
        Unscaled, dimensional output variables read via outputs[key].
    partials : partial Jacobian
        Sub-jac components written to jacobian[output_name, input_name].
    discrete_inputs : dict or None
        If not None, dict containing discrete input values.
    discrete_outputs : dict or None
        If not None, dict containing discrete output values.
    """
    deriv_vals = inst._get_jac_func()(*inst._get_compute_primal_invals(inputs, outputs,
                                                                       discrete_inputs))
    nested_tup = isinstance(deriv_vals, tuple) and len(deriv_vals) > 0 and \
        isinstance(deriv_vals[0], tuple)

    nof = len(inst._var_rel_names['output'])
    ofidx = len(inst._discrete_outputs) - 1
    for ofname in inst._var_rel_names['output']:
        ofidx += 1
        ofmeta = inst._var_rel2meta[ofname]
        for wrtidx, wrtname in enumerate(chain(inst._var_rel_names['input'],
                                               inst._var_rel_names['output'])):
            key = (ofname, wrtname)
            if key not in partials:
                # FIXME: this means that we computed a derivative that we didn't need
                continue

            wrtmeta = inst._var_rel2meta[wrtname]
            dvals = deriv_vals
            # if there's only one 'of' value, we only take the indexed value if the
            # return value of compute_primal is single entry tuple. If a single array or
            # scalar is returned, we don't apply the 'of' index.
            if nof > 1 or nested_tup:
                dvals = dvals[ofidx]

            # print(ofidx, ofname, ofmeta['shape'], wrtidx, wrtname, wrtmeta['shape'],
            #       'subjac_shape', dvals[wrtidx].shape)
            dvals = dvals[wrtidx].reshape(ofmeta['size'], wrtmeta['size'])

            sjmeta = partials.get_metadata(key)
            rows = sjmeta['rows']
            if rows is None:
                partials[ofname, wrtname] = dvals
            else:
                partials[ofname, wrtname] = dvals[rows, sjmeta['cols']]


def apply_linear(inst, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
    r"""
    Compute jac-vector product. The model is assumed to be in an unscaled state.

    If mode is:
        'fwd': (d_inputs, d_outputs) \|-> d_residuals

        'rev': d_residuals \|-> (d_inputs, d_outputs)

    Parameters
    ----------
    inst : ImplicitComponent
        The component instance.
    inputs : Vector
        Unscaled, dimensional input variables read via inputs[key].
    outputs : Vector
        Unscaled, dimensional output variables read via outputs[key].
    d_inputs : Vector
        See inputs; product must be computed only if var_name in d_inputs.
    d_outputs : Vector
        See outputs; product must be computed only if var_name in d_outputs.
    d_residuals : Vector
        See outputs.
    mode : str
        Either 'fwd' or 'rev'.
    """
    if mode == 'fwd':
        dx = tuple(chain(d_inputs.values(), d_outputs.values()))
        full_invals = tuple(inst._get_compute_primal_invals(inputs, outputs, inst._discrete_inputs))
        x = full_invals[:len(dx)]
        other = full_invals[len(dx):]
        _, deriv_vals = jax.jvp(lambda *args: inst.compute_primal(*args, *other),
                                primals=x, tangents=dx)
        if isinstance(deriv_vals, tuple):
            d_residuals.set_vals(deriv_vals)
        else:
            d_residuals.asarray()[:] = deriv_vals.flatten()
    else:
        inhash = (inputs.get_hash(), outputs.get_hash()) + tuple(inst._discrete_inputs.values())
        if inhash != inst._vjp_hash:
            # recompute vjp function only if inputs or outputs have changed
            dx = tuple(chain(d_inputs.values(), d_outputs.values()))
            full_invals = tuple(inst._get_compute_primal_invals(inputs, outputs,
                                                                inst._discrete_inputs))
            x = full_invals[:len(dx)]
            other = full_invals[len(dx):]
            _, inst._vjp_fun = jax.vjp(lambda *args: inst.compute_primal(*args, *other), *x)
            inst._vjp_hash = inhash

            if inst._compute_primals_out_shape is None:
                shape = jax.eval_shape(lambda *args: inst.compute_primal(*args, *other), *x)
                if isinstance(shape, tuple):
                    shape = (tuple(s.shape for s in shape), True, len(inst._var_rel_names['input']))
                else:
                    shape = (shape.shape, False, len(inst._var_rel_names['input']))
                inst._compute_primals_out_shape = shape

        shape, istup, ninputs = inst._compute_primals_out_shape

        if istup:
            deriv_vals = (inst._vjp_fun(tuple(d_residuals.values())))
        else:
            deriv_vals = inst._vjp_fun(tuple(d_residuals.values())[0])

        d_inputs.set_vals(deriv_vals[:ninputs])
        d_outputs.set_vals(deriv_vals[ninputs:])


def _to_compute_primal_setup_parser(parser):
    """
    Set up the command line options for the 'openmdao call_tree' command line tool.
    """
    parser.add_argument('file', nargs=1, help='Python file or module containing the class.')
    parser.add_argument('-c', '--class', action='store', dest='klass',
                        help='Component class to be converted.')
    parser.add_argument('-i', '--import', action='store', dest='imported',
                        help='Try to import the file as a module and convert the specified class.'
                        ' This requires that the class be initializable with no arguments.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help='Print status information.')
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        default='stdout', help='Output file.  Defaults to stdout.')


def _to_compute_primal_exec(options, user_args):
    """
    Process command line args and call convert on the specified class.
    """
    from openmdao.core.component import Component
    from openmdao.core.problem import Problem
    import openmdao.utils.hooks as hooks

    if not options.klass:
        raise RuntimeError("Must specify a class to convert.")

    if options.imported:
        fname = options.file[0]
        if fname.endswith('.py'):
            fname = options.file[0]
            if not os.path.exists(fname):
                raise FileNotFoundError(f"File '{fname}' not found.")

            modpath = get_module_path(fname)
            if modpath is None:
                modpath = fname
                moddir = os.path.dirname(modpath)
                sys.path = [moddir] + sys.path
                modpath = os.path.basename(modpath)[:-3]
        else:
            modpath = options.file[0]

        try:
            mod = importlib.import_module(modpath)
        except ImportError as err:
            print(f"Can't import module '{modpath}': {err}")
            return

        for name, klass in inspect.getmembers(mod, inspect.isclass):
            if name == options.klass:
                if not issubclass(klass, Component):
                    print(f"Class '{options.klass}' is not a subclass of Component.")
                    return
                # try to instantiate class with no args
                try:
                    inst = klass()
                except Exception as err:
                    print(f"Can't instantiate class '{options.klass}' with default args: {err}")
                    print("Try using --instance instead and specify the path to an instance.")
                    return

                p = Problem()
                p.model.add_subsystem('comp', inst)
                p.setup()

                to_compute_primal(inst, outfile=options.outfile)
                break
        else:
            print(f"Class '{options.klass}' not found in module '{modpath}'")
            return

    else:
        def _to_compute_primal(model):
            found = False
            classpath = options.klass.split('.')
            cname = classpath[-1]
            cmod = '.'.join(classpath[:-1])
            npaths = len(classpath)
            for s in model.system_iter(recurse=True, typ=Component):
                for cls in inspect.getmro(type(s)):
                    if cls.__name__ == cname:
                        if npaths == 1 or cls.__module__ == cmod:
                            if options.verbose:
                                print(f"Converting class '{options.klass}' compute method to "
                                      f"compute_primal method for instance '{s.pathname}'.")
                                to_compute_primal(s, outfile=options.outfile,
                                                  verbose=options.verbose)
                                found = True
                                break
                if found:
                    break
            else:
                print(f"Class '{options.klass}' not found in the model.")
                return

        def _set_dyn_hook(prob):
            # set the _to_compute_primal hook to be called right after _setup_var_data on the model
            prob.model.pathname = ''
            hooks._register_hook('_setup_var_data', class_name='Group', inst_id='',
                                 post=_to_compute_primal, exit=True)
            hooks._setup_hooks(prob.model)

        # register the hook to be called right after setup on the problem
        hooks._register_hook('setup', 'Problem', pre=_set_dyn_hook, ncalls=1)

        _load_and_exec(options.file[0], user_args)


def to_compute_primal(inst, outfile='stdout', verbose=False):
    """
    Convert the given Component's compute method to a compute_primal method that works with jax.

    Parameters
    ----------
    inst : Component
        The Component to be converted.
    outfile : str
        The name of the file to write the converted class to. Defaults to 'stdout'.
    verbose : bool
        If True, print status information.
    """
    from openmdao.core.implicitcomponent import ImplicitComponent
    from openmdao.core.explicitcomponent import ExplicitComponent

    classname = type(inst).__name__
    if verbose:
        print(f"Converting class '{classname}' compute method to compute_primal method.")
        print(f"Output will be written to '{outfile}'.")

    if isinstance(inst, ImplicitComponent):
        jaxer = ImplicitCompJaxify(inst)
    elif isinstance(inst, ExplicitComponent):
        jaxer = ExplicitCompJaxify(inst)
    else:
        print(f"'{classname}' is not an ImplicitComponent or ExplicitComponent.")
        return

    if outfile == 'stdout':
        print(jaxer.get_class_src())
    else:
        with open(outfile, 'w') as f:
            print(jaxer.get_class_src(), file=f)


if __name__ == '__main__':
    import openmdao.api as om

    def func(x, y):  # noqa: D103
        z = jnp.sin(x) * y
        q = x * 1.5
        zz = q + x * 1.5
        return z, zz

    print('partials are:\n', list(get_partials_deps(func, ('z', 'zz'))))

    p = om.Problem()
    comp = p.model.add_subsystem('comp', om.ExecComp('y = 2.0*x', x=np.ones(3), y=np.ones(3)))
    comp.derivs_method = 'jax'
    p.setup()
    p.run_model()

    print(p.compute_totals(of=['comp.y'], wrt=['comp.x']))
