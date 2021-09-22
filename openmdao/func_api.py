"""
API to associate metadata with and retrieve metadata from function objects.
"""

from numbers import Number
import ast
import inspect
import textwrap
import warnings
import numpy as np
from contextlib import contextmanager
from functools import wraps

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


_allowed_add_input_args = {
    'val', 'shape', 'src_indices', 'flat_src_indices', 'units', 'desc', 'tags', 'shape_by_conn',
    'copy_shape', 'distributed', 'new_style_idx',
}

_allowed_add_output_args = {
    'val', 'shape', 'units', 'res_units', 'desc' 'lower', 'upper', 'ref', 'ref0', 'res_ref', 'tags',
    'shape_by_conn', 'copy_shape', 'distributed',
}

_allowed_add_var_args = _allowed_add_input_args.union(_allowed_add_output_args)


class OMWrappedFunc(object):
    """
    Function wrapper that holds function metadata useful to OpenMDAO.

    Parameters
    ----------
    func : function
        The function to be wrapped.

    Attributes
    ----------
    _f : function
        The wrapped function.
    _defaults : dict
        Dict of default metadata values that could apply to any variable.
    _metadata : dict
        Dict of metadata values that must apply to all variables.
    _inputs : dict
        Dict of metadata dicts keyed to input name.
    _outputs : dict
        Dict of metadata dicts keyed to output name.
    _declare_partials : list
        List of keyword args, one entry for each call to declare_partials.
    _declare_colorings : list
        List of keyword args, one entry for each call to declare_coloring.
    _call_setup : bool
        If True, call the setup functions for input and output metadata.
    _use_jax : bool
        If True, use jax to compute output shapes based on input shapes.
    """

    def __init__(self, func):
        self._f = func
        self._defaults = {'val': 1.0, 'shape': ()}
        self._metadata = {}

        # populate _inputs dict with input names based on function signature so we can error
        # check vs. inputs added via add_input
        self._inputs = {n: {'val': None if p.default is inspect._empty else p.default}
                        for n, p in inspect.signature(func).parameters.items()}
        self._outputs = {}
        self._declare_partials = []
        self._declare_colorings = []
        self._call_setup = True
        self._use_jax = False

    def __call__(self, *args, **kwargs):
        r"""
        Call the wrapped function.

        Parameters
        ----------
        *args : list
            Positional args.
        **kwargs : dict
            Keyword args.

        Returns
        -------
        object
            The return of the wrapped function.
        """
        return self._f(*args, **kwargs)

    def defaults(self, **kwargs):
        r"""
        Add metadata that may apply to any inputs or outputs of the wrapped function.

        Any variable specific metadata will override any metadata specified here.

        Parameters
        ----------
        **kwargs : dict
            Metadata names and their values.
        """
        self._defaults.update(kwargs)
        return self

    def metadata(self, **kwargs):
        r"""
        Add metadata that applies to all variables of the wrapped function.

        If a variable has specific metadata that doesn't match any metadata specified here,
        an exception will be raised.

        Parameters
        ----------
        **kwargs : dict
            Metadata names and their values.
        """
        self._metadata.update(kwargs)
        return self

    def add_input(self, name, **kwargs):
        r"""
        Add metadata for an input of the wrapped function.

        Parameters
        ----------
        name : str
            Name of the input variable.
        **kwargs : dict
            Keyword args to store.
        """
        if name not in self._inputs:
            raise NameError(f"'{name}' is not an input to this function.")
        meta = self._inputs[name]
        for kw in kwargs:
            if kw in meta and meta[kw] is not None:
                raise RuntimeError("Metadata has already been added to function for input "
                                   f"'{name}'.")
        if meta.get('val') is not None and kwargs.get('val') is not None:
            self._check_vals_equal(name, meta['val'], kwargs['val'])
        meta.update(kwargs)
        return self

    def add_inputs(self, *kwargs):
        r"""
        Add metadata for multiple inputs of the wrapped function.

        Parameters
        ----------
        **kwargs : dict
            Keyword args to store.  The value corresponding to each key is a dict containing the
            metadata for the input name that matches that key.
        """
        for name, meta in kwargs.items():
            self.add_input(name, meta)
        return self

    def add_output(self, name, **kwargs):
        r"""
        Add metadata for an output of the wrapped function.

        Parameters
        ----------
        name : str
            Name of the output variable.
        **kwargs : dict
            Keyword args to store.
        """
        if name in self._inputs:
            raise RuntimeError(f"'{name}' already registered as an input")
        if name in self._outputs:
            raise RuntimeError(f"'{name}' already registered as an output")
        self._outputs[name] = kwargs
        return self

    def add_outputs(self, *kwargs):
        r"""
        Add metadata for multiple outputs of the wrapped function.

        Parameters
        ----------
        **kwargs : dict
            Keyword args to store.  The value corresponding to each key is a dict containing the
            metadata for the output name that matches that key.
        """
        for name, meta in kwargs.items():
            self.add_output(name, meta)
        return self

    def output_names(self, names):
        """
        Set the names of a function's output variables.

        Parameters
        ----------
        names : list of str
            Names of outputs with order matching order of return values.

        Returns
        -------
        function
            A function wrapper that updates the function's metadata.
        """
        kwargs = {n: {} for n in names}
        return self.add_outputs(**kwargs)

    def declare_option(self, name, **kwargs):
        r"""
        Collect name and keyword args to later declare an option on an OpenMDAO component.

        Parameters
        ----------
        name : str
            Name of the option variable.
        **kwargs : dict
            Keyword args to store.
        """
        self._inputs[name].update(kwargs)
        self._inputs[name]['is_option'] = True

    def declare_partials(self, **kwargs):
        r"""
        Collect args to be passed to declare_partials on an OpenMDAO component.

        Parameters
        ----------
        **kwargs : dict
            Keyword args to store.
        """
        self._declare_partials.append(kwargs)
        if 'method' in kwargs and kwargs['method'] == 'jax':
            self._use_jax = True

    def declare_coloring(self, **kwargs):
        r"""
        Collect args to be passed to declare_coloring on an OpenMDAO component.

        Parameters
        ----------
        **kwargs : dict
            Keyword args to store.
        """
        self._declare_colorings.append(kwargs)

    def get_input_meta(self):
        """
        Get an iterator of (name, metdata_dict) for each input variable.

        Returns
        -------
        iter of (str, dict)
            Iterator of (name, metdata_dict) for each input variable.
        """
        if self._call_setup:
            self._setup()
        return list(self._inputs.items())

    def get_output_meta(self):
        """
        Get an iterator of (name, metdata_dict) for each output variable.

        Returns
        -------
        iter of (str, dict)
            Iterator of (name, metdata_dict) for each output variable.
        """
        if self._call_setup:
            self._setup()
        return self._outputs.items()

    def get_declare_partials(self):
        """
        Get an iterator of keyword args passed to each declare_partials decorator call.

        Returns
        -------
        iter of dict
            Iterator of dicts containing the keyword args for each call.
        """
        return self._declare_partials.items()

    def get_declare_colorings(self):
        """
        Get an iterator of keyword args passed to each declare_coloring decorator call.

        Returns
        -------
        iter of dict
            Iterator of dicts containing the keyword args for each call.
        """
        return self._declare_coloring.items()

    def _check_vals_equal(self, name, val1, val2):
        """
        Compare two values that could be a mix of ndarray and other types.

        Parameters
        ----------
        name : str
            Name of the variable (for error reporting).
        val1 : object
            First value.
        val2 : object
            Second value.
        """
        # == is more prone to raise exceptions when ndarrays are involved, so use !=
        neq = val1 != val2
        if (isinstance(neq, np.ndarray) and np.any(neq)) or neq:
            raise RuntimeError(f"Conflicting metadata entries for '{name}'.")

    def _resolve_meta(self, key, meta):
        """
        Update the value of the metadata corresponding to key based on self._metadata.

        Parameters
        ----------
        key : str
            The metadata entry key.
        meta : dict
            The metadata dict to be updated.
        """
        if key in self._metadata:
            mval = self._metadata[key]
            if key in meta:
                val = meta[key]
                # check for conflict with func metadata
                if val is None:
                    meta[key] = mval
                else:
                    self._check_vals_equal(key, val, mval)
            else:
                meta[key] = mval

    def _resolve_default(self, key, meta):
        """
        Update the value of the metadata corresponding to key based on self._defaults.

        Parameters
        ----------
        key : str
            The metadata entry key.
        meta : dict
            The metadata dict to be updated.
        """
        if (key not in meta or meta[key] is None) and key in self._defaults:
            meta[key] = self._defaults[key]

    def _setup(self):
        """
        Set up input and output variable metadata dicts.
        """
        self._call_setup = False
        overrides = set(self._defaults)
        overrides.update(self._metadata)

        self._setup_inputs(overrides)
        self._setup_outputs(overrides)

    def _setup_inputs(self, overrides):
        """
        Set up the input variable metadata dicts.

        Parameters
        ----------
        overrides : set
            Set of names of entries in self._defaults and self._metadata.
        """
        ins = self._inputs
        overrides = overrides - {'val', 'shape'}

        # first, retrieve inputs from the function signature
        for name in inspect.signature(self._f).parameters:
            meta = ins[name]
            if meta.get('is_option'):
                continue

            # set using defaults or metadata if val has not been set
            self._resolve_meta('val', meta)

            if 'val' in meta and meta['val'] is not None:
                valshape = np.asarray(meta['val']).shape
            else:
                valshape = None
                meta['val'] = self._defaults['val']

            self._resolve_meta('shape', meta)

            if meta.get('shape') is None:
                if valshape is not None:
                    meta['shape'] = valshape
                else:
                    meta['shape'] = self._defaults['shape']

            meta['shape'] = _shape2tuple(meta['shape'])
            if not valshape:  # val is a scalar so reshape with the given meta['shape']
                meta['val'] = np.ones(meta['shape']) * meta['val']
            elif valshape != meta['shape']:
                raise ValueError(f"Input '{name}' default value has shape "
                                 f"{valshape}, but shape was specified as {meta['shape']}.")

            for o in overrides:
                self._resolve_meta(o, meta)
                self._resolve_default(o, meta)

    def _setup_outputs(self, overrides):
        """
        Set up the output variable metadata dicts.

        Parameters
        ----------
        overrides : set
            Set of names of entries in self._defaults and self._metadata.
        """
        outmeta = {}

        # Parse the function code to possibly identify the names of the return values and
        # input/output dependencies. Return names will be non-None only if they are a simple name,
        # e.g., return a, b, c
        outlist = []
        try:
            ret_info = get_function_deps(self._f)
        except RuntimeError:
            #  this could happen if function is compiled or has multiple return lines
            if not self._outputs:
                raise RuntimeError(f"Couldn't determine function return names or "
                                   "number of return values based on AST and no return value "
                                   "annotations were supplied.")
            warnings.warn("Couldn't determine function return names based on AST.  Assuming number "
                          "of return values matches number of return value annotations.")
            outlist = list(self._outputs.items())
        else:
            for o, deps in ret_info:
                outlist.append([o, {'deps': deps}])

        notfound = []
        for oname, ometa in self._outputs.items():
            for n, meta in outlist:
                if n == oname:
                    if meta is not ometa:
                        meta.update(ometa)
                    break
            else:  # didn't find oname
                notfound.append(oname)

        if notfound:  # try to fill in the unnamed slots with annotated output data
            inones = [i for i, (n, m) in enumerate(outlist) if n is None]  # indices with no name
            if len(notfound) != len(inones):
                raise RuntimeError(f"Number of unnamed return values "
                                   f"({len(inones)}) doesn't match number of unmatched annotated "
                                   f"return values ({len(notfound)}).")

            # number of None return slots equals number of annotated entries not found in outlist
            for i_olist, name_notfound in zip(inones, notfound):
                annotated_meta = outmeta[name_notfound]
                _, ret_meta = outlist[i_olist]
                ret_meta.update(annotated_meta)
                outlist[i_olist] = (name_notfound, ret_meta)

        outs = {n: m for n, m in outlist}

        self._compute_out_shapes(self._inputs, outs)

        for meta in outs.values():
            for o in overrides:
                self._resolve_meta(o, meta)
                self._resolve_default(o, meta)

        self._outputs = outs

    def _compute_out_shapes(self, ins, outs):
        """
        Compute the shapes of outputs based on those of the inputs.

        Parameters
        ----------
        ins : dict
            Dict of input metadata containing input shapes.
        outs : dict
            Dict of output metadata that will be updated with shape information.
        """
        need_shape = []
        for name, ometa in outs.items():
            try:
                ometa['shape']
            except KeyError:
                need_shape.append(name)

        args = []
        for name, meta in ins.items():
            if meta['val'] is not None:
                args.append(meta['val'])
            else:
                try:
                    shp = meta['shape']
                except KeyError:
                    raise RuntimeError(f"Can't determine shape of input '{name}'.")
                if jax is not None:
                    args.append(jax.ShapedArray(_shape2tuple(shp), dtype=np.float64))

        # compute shapes as a check against annotated value (if any)
        if jax is not None and self._use_jax:
            # must replace numpy with jax numpy when making jaxpr.
            with jax_context(self._f.__globals__):
                try:
                    v = jax.make_jaxpr(self._f)(*args)
                except Exception as err:
                    if need_shape:
                        raise RuntimeError(f"Failed to determine the output shapes "
                                           f"based on the input shapes. The error was: {err}.  To "
                                           "avoid this error, add return value annotations that "
                                           "specify the shapes of the return values to the "
                                           "function.")
                    warnings.warn("Failed to determine the output shapes based on the input "
                                  "shapes in order to check the provided annotated values. The"
                                  f" error was: {err}.")
                else:
                    for val, name in zip(v.out_avals, outs):
                        oldshape = outs[name].get('shape')
                        if oldshape is not None and _shape2tuple(oldshape) != val.shape:
                            raise RuntimeError(f"Annotated shape for return value "
                                               f"'{name}' of {oldshape} doesn't match computed "
                                               f"shape of {val.shape}.")
                        outs[name]['shape'] = val.shape
                    need_shape = []

        if need_shape:  # output shapes weren't provided by user or by jax
            if 'shape' in self._metadata:
                shape = self._metadata['shape']
            else:
                shape = self._defaults['shape']
            warnings.warn(f"Return values {need_shape} have unspecified shape so are assumed to "
                          f"have shape {shape}.")
            for name in need_shape:
                outs[name]['shape'] = shape


def wrap(func):
    """
    Return a wrapped function object.

    If arg is already a wrapped function object, return that.

    Parameters
    ----------
    func : function or OMwrappedFunc
        A plain or already wrapped function object.

    Returns
    -------
    OMwrappedFunc
        The wrapped function object.
    """
    if isinstance(func, OMWrappedFunc):
        return func
    return OMWrappedFunc(func)


def _get_kwargs(func, locals_dict, default=None):
    """
    Convert a function's args to a kwargs dict containing entries that are not identically default.

    Parameters
    ----------
    func : function
        The function whose args we want to convert to kwargs.
    locals_dict : dict
        The locals dict for the function.
    default : object
        Don't include arguments whose values are this object.

    Returns
    -------
    dict
        The non-default keyword args dict.
    """
    return {n: locals_dict[n] for n in inspect.signature(func).parameters
            if locals_dict[n] is not default}


def _check_kwargs(kwargs, allowed, fname):
    """
    Check contents of kwargs for args that aren't allowed.

    Parameters
    ----------
    kwargs : dict
        Original keyword args dict.
    allowed : set
        Set of allowed arg names.
    fname : str
        Function name (for error reporting).
    """
    errs = [n for n in kwargs if n not in allowed]
    if errs:
        raise RuntimeError(f"The following args passed to {fname} are not allowed: {errs}.")


def _shape2tuple(shape):
    """
    Return shape as a tuple.

    Parameters
    ----------
    shape : int or tuple
        The given shape.

    Returns
    -------
    tuple
        The shape as a tuple.
    """
    if isinstance(shape, Number):
        return (shape,)
    elif shape is None:
        return shape
    return tuple(shape)


@contextmanager
def jax_context(globals):
    """
    A context where np and numpy are replaced by their jax equivalents.

    Parameters
    ----------
    globals : dict
        The globals dict to have its numpy/np attributes updated.
    """
    savenp = savenumpy = None
    if 'np' in globals and globals['np'] is np:
        savenp = globals['np']
        globals['np'] = jnp
    if 'numpy' in globals:
        savenumpy = globals['numpy']
        globals['numpy'] = jnp
    try:
        yield
    finally:
        if savenp is not None:
            globals['np'] = savenp
        if savenumpy is not None:
            globals['numpy'] = savenumpy


def _get_long_name(node):
    """
    Return a name (possibly dotted) corresponding to the give node or None.

    If the node is a Name node or an Attribute node that is composed only of other Attribute or
    Name nodes, then return the full dotted name for this node. Otherwise, i.e., if this node
    contains other expressions.

    Parameters
    ----------
    node : ASTnode
        A node of an abstract syntax tree.

    Returns
    -------
    str or None
        Name corresponding to the given node.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif not isinstance(node, ast.Attribute):
        return None
    val = node.value
    parts = [node.attr]
    while True:
        if isinstance(val, ast.Attribute):
            parts.append(val.attr)
            val = val.value
        elif isinstance(val, ast.Name):
            parts.append(val.id)
            break
        else:  # it's more than just a simple dotted name
            return None
    return '.'.join(parts[::-1])


class _FuncDepCollector(ast.NodeVisitor):
    """
    An ast.NodeVisitor that records dependencies between inputs and outputs.

    Each instance of this is single-use.  If needed multiple times create a new instance
    each time.  It also assumes that the AST to be visited contains only a single function
    definition.

    Attributes
    ----------
    _ret_info : list
        List containing name (or None) for each function return value.
    """

    def __init__(self, func):
        super().__init__()
        self._attrs = None
        self._deps = {}
        self._ret_info = []
        self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec'))

    def _do_assign(self, targets, rhs):
        lhs_attrs = []
        for t in targets:
            lhs_attrs.append(_get_long_name(t))

        self._attrs = set()
        self.visit(rhs)

        for a in lhs_attrs:
            if a not in self._deps:
                self._deps[a] = set()
            self._deps[a].update(self._attrs)

        self._attrs = None

    def visit_Attribute(self, node):
        if self._attrs is not None:
            self._attrs.add(_get_long_name(node))

    def visit_Name(self, node):
        if self._attrs is not None:
            self._attrs.add(node.id)

    def visit_Assign(self, node):
        self._do_assign(node.targets, node.value)

    def visit_AugAssign(self, node):
        self._do_assign((node.target,), node.value)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._do_assign((node.target,), node.value)

    def visit_Call(self, node):  # (func, args, keywords, starargs, kwargs)
        for arg in node.args:
            self.visit(arg)

        for kw in node.keywords:
            self.visit(kw.value)

    def _get_return_attrs(self, node):
        self._attrs = set()

        self.visit(node)
        # also include a boolean indicating if the return expr is a simple name
        self._ret_info.append((tuple(self._attrs), isinstance(node, ast.Name)))

        self._attrs = None

    def visit_Return(self, node):
        """
        Visit a Return node.

        Parameters
        ----------
        node : ASTnode
            The return node being visited.
        """
        if self._ret_info:
            raise RuntimeError("_FuncDepCollector does not support multiple returns in a "
                               "single function.  Either the given function contains multiple "
                               "returns or this _FuncDepCollector instance has been used "
                               "more than once, which is unsupported.")

        if isinstance(node.value, ast.Tuple):
            for n in node.value.elts:
                self._get_return_attrs(n)
        else:
            self._get_return_attrs(node.value)


def get_function_deps(func):
    """
    Return dependency between return value(s) and inputs.

    Parameters
    ----------
    func : function
        The function used to compute input/output dependencies.

    Returns
    -------
    list
        List of the form (name or None, dependency_set) containing one entry for each return
        value.  'name' will be the name of the return value if it has a simple name, otherwise
        None.
    """
    input_names = set(inspect.signature(func).parameters)
    funcdeps = _FuncDepCollector(func)
    deps = funcdeps._deps
    retdeps = []
    for names, _ in funcdeps._ret_info:
        depset = set()
        for n in names:
            stack = [n]
            seen = set()
            while stack:
                v = stack.pop()
                seen.add(v)
                if v in input_names:
                    depset.add(v)
                elif v in deps:
                    stack.extend([d for d in deps[v] if d not in seen])

        retdeps.append(depset)

    return [(n[0] if simple else None, d) for ((n, simple), d) in zip(funcdeps._ret_info, retdeps)]
