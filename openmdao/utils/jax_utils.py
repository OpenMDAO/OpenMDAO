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
from scipy.sparse import coo_matrix

from openmdao.utils.code_utils import _get_long_name, remove_src_blocks, replace_src_block, \
    get_function_deps
from openmdao.utils.file_utils import get_module_path, _load_and_exec
from openmdao.utils.om_warnings import issue_warning


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
        # variable or some other expression, then we don't modify it and the conversion will
        # likely fail.
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
    _fstack : list
        The stack of function definitions being visited.
    """

    def __init__(self, method):  # noqa
        self._returns = []
        self._fstack = []
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

    def visit_FunctionDef(self, node):
        """
        Visit a FunctionDef node.

        Parameters
        ----------
        node : ASTnode
            The function definition node being visited.
        """
        if self._fstack:
            return  # skip nested functions
        self._fstack.append(node)
        for stmt in node.body:
            self.visit(stmt)
        self._fstack.pop()


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


if jax is None:
    def _jax_register_pytree_class(cls):
        pass

else:

    _registered_classes = set()

    def _jax_register_pytree_class(cls):
        """
        Register a class with jax so that it can be used with jax.jit.

        This can be called after instantiating the class if necessary.

        Parameters
        ----------
        cls : class
            The class to be registered.
        """
        global _registered_classes
        if cls not in _registered_classes:
            # register with jax so we can flatten/unflatten self
            tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)
            _registered_classes.add(cls)


def get_vmap_tangents(vals, direction, fill=1., coloring=None):
    """
    Return a tuple of tangents values for use with vmap.

    The batching dimension is the last axis of each tangent.

    Parameters
    ----------
    vals : list
        List of function input or output values.
    direction : str
        The direction to compute the sparsity in.  It must be 'fwd' or 'rev'.
    fill : float
        The value to fill nonzero entries in the tangent with.
    coloring : Coloring or None
        A Coloring object that contains coloring information including nonzero indices.

    Returns
    -------
    tuple of ndarray or ndarray
        The tangents values to be passed to vmap.
    """
    sizes = [np.size(a) for a in vals]
    totsize = np.sum(sizes)

    if coloring is None:
        # start with a full diagonal matrix, which allows us to set a seed for each input value
        # in parallel.
        arr = np.empty(totsize)
        arr[:] = fill
        tangent = np.diag(arr)
        ncols = totsize
    else:
        # using coloring, so 'compress' the diagonal matrix to one with ncolors columns.
        # columns are the batching dimension for vmap and each column also corresponds to a color.
        colors = list(coloring.color_iter(direction))
        tangent = np.zeros((totsize, len(colors)))
        for i, nzs in enumerate(colors):
            tangent[nzs, i] = 1.
        ncols = len(colors)
    # take the 2D tangent array and reshape it to match the shape of each input variable.
    # (with the additional batching dimension as the last axis)
    tangents = []
    start = end = 0
    for v in vals:
        end += np.size(v)
        tangents.append(jnp.array(tangent[start:end].reshape(np.shape(v) + (ncols,))))
        start = end

    tangents = tuple(tangents)

    return tangents


def _update_subjac_sparsity(sparsity_iter, pathname, subjacs_info):
    """
    Update subjac sparsity info based on the given sparsity iterator.

    Parameters
    ----------
    sparsity_iter : iter of tuple
        Tuple of the form (of, wrt, rows, cols, shape).
    pathname : str
        The pathname of the component.
    subjacs_info : dict
        The subjac sparsity info.
    """
    prefix = pathname + '.'
    for of, wrt, rows, cols, shape in sparsity_iter:
        # sparsity uses relative names, so convert to absolute
        abs_key = (prefix + of, prefix + wrt)
        if abs_key not in subjacs_info:
            if rows is not None and len(rows) == 0:
                continue

            subjacs_info[abs_key] = {
                'shape': shape,
                'dependent': True,
                'rows': rows,
                'cols': cols,
                'diagonal': False,
                'val': np.zeros(shape) if rows is None else np.zeros(len(rows))
            }
        else:
            subj = subjacs_info[abs_key]
            diag = subj['diagonal']
            if diag:
                assert shape[0] == shape[1]
                if rows is None:
                    raise RuntimeError(f"Subjacobian ({of}, {wrt}) is labeled as diagonal but is "
                                       "dense.")
                elif len(rows) > shape[0]:
                    raise RuntimeError(f"Subjacobian ({of}, {wrt}) is labeled as diagonal but the "
                                       "number of row/cols > diag size "
                                       f"({len(rows)} > {shape[0]}).")
                elif len(rows) < shape[0]:
                    subj['diagonal'] = diag = False
                    issue_warning(f"Subjacobian ({of}, {wrt}) is labeled as diagonal but is "
                                  "actually more sparse than that, row/cols < diag size "
                                  f"({len(rows)} < {shape[0]}).")

            if rows is not None:
                if len(rows) == 0:
                    del subjacs_info[abs_key]
                else:
                    subj['sparsity'] = (rows, cols, shape)


def _re_init(self):
    """
    Re-initialize the component for a new run.
    """
    self._tangents = {'fwd': None, 'rev': None}
    self._do_sparsity = False
    self._sparsity = None
    self._jac_func_ = None
    self._static_hash = None
    self._jac_colored_ = None
    self._output_shapes = None
    self._do_shape_check = True


def _get_differentiable_compute_primal(self, discrete_inputs):
    """
    Get the compute_primal function for the jacobian.

    This version of the compute primal should take no discrete inputs and return no discrete
    outputs. It will be called when computing the jacobian.

    Parameters
    ----------
    self : Component
        The component to get the compute_primal function for.
    discrete_inputs : iter of discrete values
        The discrete input values.

    Returns
    -------
    function
        The compute_primal function to be used to compute the jacobian.
    """
    # exclude the discrete inputs from the inputs and the discrete outputs from the outputs
    if discrete_inputs:
        if self._discrete_outputs:
            ncontouts = self._outputs.nvars()

            def differentiable_compute_primal(*contvals):
                return self.compute_primal(*contvals, *discrete_inputs)[:ncontouts]

        else:

            def differentiable_compute_primal(*contvals):
                return self.compute_primal(*contvals, *discrete_inputs)

        return differentiable_compute_primal

    elif self._discrete_outputs:
        ncontouts = self._outputs.nvars()

        def differentiable_compute_primal(*contvals):
            return self.compute_primal(*contvals)[:ncontouts]

        return differentiable_compute_primal

    return self.compute_primal


def _compute_sparsity(self, direction=None, num_iters=1, perturb_size=1e-9, use_nan=False):
    """
    Compute the sparsity of the Jacobian using jvp/vjp with nans for the seeds.

    Parameters
    ----------
    self : Component
        The component to compute the sparsity for.
    direction : str or None
        The direction to compute the sparsity in.  If None, the best direction is chosen based
        on the number of inputs and outputs.  If a str, it must be 'fwd' or 'rev'.
    num_iters : int
        The number of times to run the perturbation iteration.
    perturb_size : float
        The size of the perturbation to use.
    use_nan : bool
        If True, use nans for the seeds.

    Returns
    -------
    coo_matrix, dict
        The boolean sparsity matrix and info.
    """
    if direction is None:
        direction = self.best_partial_deriv_direction()

    assert direction in ['fwd', 'rev']

    ncontouts = self._outputs.nvars()

    implicit = not self.is_explicit(is_comp=True)

    if implicit:
        ncontins = self._inputs.nvars() + ncontouts
        pvecs = (self._outputs, self._inputs)
        wrtsize = len(self._outputs) + len(self._inputs)
        save_vecs = (self._residuals,)
    else:
        ncontins = self._inputs.nvars()
        pvecs = (self._inputs,)
        wrtsize = len(self._inputs)
        save_vecs = (self._outputs, self._residuals,)

    sparsity = None
    idiscvals = tuple(self._discrete_inputs.values())

    # exclude the discrete inputs from the inputs and the discrete outputs from the outputs
    differentiable_part = _get_differentiable_compute_primal(self, idiscvals)

    # when computing tangents we only care about shapes of the values, not the values themselves,
    # so we can use the unperturbed values for the tangents
    full_invals = tuple(self._get_compute_primal_invals())
    icontvals = full_invals[:ncontins]  # continuous inputs
    if direction == 'fwd':
        tangents = get_vmap_tangents(icontvals, 'fwd', fill=np.nan if use_nan else 1.)

        def jvp_at_point(tangent, contvals):
            # [1] is the derivative, [0] is the primal (we don't need the primal)
            return jax.jvp(differentiable_part, contvals, tangent)[1]

        Jfunc = jax.vmap(jvp_at_point, in_axes=[-1, None], out_axes=-1)
    else:
        # these are really cotangents
        tangents = get_vmap_tangents(tuple(self._outputs.values()), 'rev',
                                     fill=np.nan if use_nan else 1.)

        def vjp_at_point(cotangent, contvals):
            return jax.vjp(differentiable_part, *contvals)[1](cotangent)

        # vectorize over last axis of cotangents
        Jfunc = jax.vmap(vjp_at_point, in_axes=[-1, None], out_axes=-1)

    if self.options['use_jit']:
        Jfunc = jax.jit(Jfunc)

    sparsity = np.zeros((len(self._outputs), wrtsize))

    for _ in self._perturbation_iter(num_iters=num_iters, perturb_size=perturb_size,
                                     perturb_vecs=pvecs, save_vecs=save_vecs):
        self._apply_nonlinear()

        full_invals = tuple(self._get_compute_primal_invals())

        J = Jfunc(tangents, full_invals[:ncontins])  # :ncontins are the continuous inputs

        if not isinstance(J, tuple):
            J = (J,)

        if len(J) == 1:
            J = J[0]
            if len(J.shape) > 2:
                # flatten 'variable' dimensions.  Last dimension is the batching dimension.
                J = J.reshape(np.prod(J.shape[:-1], dtype=int), J.shape[-1])
            elif len(J.shape) == 1:
                J = np.atleast_2d(J)
        else:
            # flatten 'variable' dimensions for each variable.  Last dimension is the batching
            # dimension.  Then vertically stack all the flattened 'variable' arrays.
            J = np.vstack([j.reshape(np.prod(j.shape[:-1], dtype=int), j.shape[-1]) for j in J])

        if direction != 'fwd':
            J = J.T

        if sparsity is None:
            sparsity[:, :] = np.abs(J)
        else:
            sparsity[:, :] += np.abs(J)

    if implicit:
        # we need to swap input and output cols because OpenMDAO jacs have output wrts first
        # followed by input wrts but compute_primal takes inputs first followed by outputs
        sparsity = np.hstack((sparsity[:, -len(self._outputs):], sparsity[:, :len(self._inputs)]))

    nz = np.nonzero(sparsity)
    data = np.ones(len(nz[0]), dtype=bool)
    sparsity = coo_matrix((data, nz), shape=sparsity.shape)

    info = {
        'tol': 0.,
        'orders': None,
        'good_tol': 0.,
        'nz_matches': 0,
        'n_tested': 0,
        'nz_entries': len(nz[0]),
        'J_shape': sparsity.shape,
    }

    self._update_subjac_sparsity(self.subjac_sparsity_iter(sparsity=sparsity))

    return sparsity, info


def _compute_output_shapes(func, input_shapes):
    """
    Compute the shapes of the outputs of the function.

    The function must be traceable by jax.

    Parameters
    ----------
    func : function
        The function to compute the output shapes for.
    input_shapes : list
        The shapes of the input variables, or None if the input isn't a scalar or array.
    """
    argnames = list(inspect.signature(func).parameters)
    traceargs = []
    for argname in argnames:
        inshape = input_shapes.get(argname)
        if inshape is not None:
            traceargs.append(jax.ShapeDtypeStruct(inshape, jnp.float64))
        else:
            traceargs.append(None)

    retvals = jax.eval_shape(func, *traceargs)
    if not isinstance(retvals, tuple):
        retvals = (retvals,)

    retshapes = []
    for val in retvals:
        try:
            retshapes.append(val.shape)
        except AttributeError:
            retshapes.append(None)

    return retshapes


def _ensure_returns_tuple(func):
    """
    Ensure that the function returns a tuple.

    If the function already returns a tuple, it is returned unchanged.
    Otherwise, a wrapper function is returned that returns a tuple.

    If for some reason the function cannot be parsed, it is returned unchanged.

    Parameters
    ----------
    func : function
        The function to ensure returns a tuple.

    Returns
    -------
    function
        The function that returns a tuple.
    """
    try:
        checker = ReturnChecker(func)
    except Exception:
        issue_warning(f"Failed to parse function {func.__name__} to check if it returns a tuple."
                      "Returning original function.")
        return func
    else:
        if checker.returns_tuple():
            return func
        else:
            def wrapper(*args, **kwargs):
                return (func(*args, **kwargs),)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper


def _jax2np(J):
    """
    Take the return of vmapped jvp/vjp and convert to a numpy array.

    Parameters
    ----------
    J : tuple or jax array
        The return of vmapped jvp/vjp.

    Returns
    -------
    ndarray
        The numpy array.
    """
    if isinstance(J, tuple):
        if len(J) == 1:
            J = np.asarray(J[0])
            # reshape(-1, ...) to flatten all but the last dimension
            return J.reshape(-1, J.shape[-1])
        else:
            return np.concatenate([np.asarray(a).reshape(-1, a.shape[-1]) for a in J])
    else:
        return np.asarray(J).reshape(-1, J.shape[-1])


def _jax_derivs2partials(self, deriv_vals, partials, ofnames, wrtnames):
    """
    Copy JAX derivatives into partials.

    Parameters
    ----------
    self : Component
        The component to copy the derivatives into.
    deriv_vals : tuple
        The derivatives.
    partials : dict
        The partials to copy the derivatives into, keyed by (of_name, wrt_name).
    ofnames : list
        The output names.
    wrtnames : list
        The input names.
    """
    nested_tup = isinstance(deriv_vals, tuple) and len(deriv_vals) > 0 and \
        isinstance(deriv_vals[0], tuple)
    nof = len(ofnames)

    wrtnames = list(wrtnames)
    for ofidx, ofname in enumerate(ofnames):
        ofmeta = self._var_rel2meta[ofname]
        for wrtidx, wrtname in enumerate(wrtnames):
            key = (ofname, wrtname)
            if key not in partials:
                # FIXME: this means that we computed a derivative that we didn't need
                continue

            dvals = deriv_vals
            # if there's only one 'of' value, we only take the indexed value if the
            # return value of compute_primal is single entry tuple. If a single array or
            # scalar is returned, we don't apply the 'of' index.
            if nof > 1 or nested_tup:
                dvals = dvals[ofidx]

            dvals = dvals[wrtidx].reshape(ofmeta['size'], self._var_rel2meta[wrtname]['size'])

            sjmeta = partials.get_metadata(key)
            rows = sjmeta['rows']
            if rows is None:
                partials[ofname, wrtname] = dvals
            else:
                partials[ofname, wrtname] = dvals[rows, sjmeta['cols']]


def _check_output_shapes(self):
    """
    Compare declared output shapes of compute_primal to those computed by jax.eval_shape.
    """
    rel_vars = self._var_rel2meta
    output_shapes = {n: rel_vars[n]['shape'] for n in self._var_rel_names['output']}

    discrete_inputs = self._discrete_inputs.values() if self._discrete_inputs else ()
    differentiable_cp = _get_differentiable_compute_primal(self, discrete_inputs)

    tracing_args = self._get_compute_primal_tracing_args()
    retvals = jax.eval_shape(differentiable_cp, *tracing_args)
    if not isinstance(retvals, tuple):
        retvals = (retvals,)

    computed_out_shapes = []
    for val in retvals:
        try:
            oshape = val.shape
            if oshape == (0,):
                oshape = ()
            computed_out_shapes.append(oshape)
        except AttributeError:
            computed_out_shapes.append(None)

    if len(output_shapes) != len(computed_out_shapes):
        raise RuntimeError(f"{self.msginfo}: The number of continuous outputs returned from "
                           f"compute_primal ({len(computed_out_shapes)}) doesn't match the number "
                           f"of declared continuous outputs ({len(output_shapes)}).")

    bad = []
    for shape_tup, comp_shape in zip(output_shapes.items(), computed_out_shapes):
        name, shape = shape_tup
        if shape != comp_shape:
            bad.append(f"Shape mismatch for output '{name}': expected {shape} but "
                       f"got {comp_shape}.")

    if bad:
        msg =  '\n   '.join(bad)
        raise RuntimeError(f"{self.msginfo}:\n   {msg}")


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


def _update_add_input_kwargs(self, **kwargs):
    if self.options['default_to_dyn_shapes']:
        if kwargs.get('val') is None and kwargs.get('shape') is None:
            if kwargs.get('copy_shape') is None and kwargs.get('compute_shape') is None:
                if kwargs.get('shape_by_conn') is None:
                    kwargs['shape_by_conn'] = True

    return kwargs


def _update_add_output_kwargs(self, name, **kwargs):
    if self.options['default_to_dyn_shapes']:
        if kwargs.get('val') is None and kwargs.get('shape') is None:
            if kwargs.get('copy_shape') is None and kwargs.get('compute_shape') is None:
                # add our own compute_shape function
                kwargs['compute_shape'] = self._get_compute_shape_func(name)

    return kwargs


if __name__ == '__main__':
    import openmdao.api as om

    def func(x, y):  # noqa: D103
        z = jnp.sin(x) * y
        q = x * 1.5
        zz = q + x * 1.5
        return z, zz

    print('partials are:\n', list(get_function_deps(func, ('z', 'zz'))))

    p = om.Problem()
    comp = p.model.add_subsystem('comp', om.ExecComp('y = 2.0*x', x=np.ones(3), y=np.ones(3)))
    comp.derivs_method = 'jax'
    p.setup()
    p.run_model()

    print(p.compute_totals(of=['comp.y'], wrt=['comp.x']))
