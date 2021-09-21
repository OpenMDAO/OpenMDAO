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

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


#
# User API (decorators used to associate metadata with the function)
#

def add_input(name, **kwargs):
    """
    Set metadata associated with one of a function's input variables.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).add_input(name, **kwargs)
    return _wrap


def add_output(name, **kwargs):
    """
    Set metadata associated with one of a function's return values.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).add_output(name, **kwargs)
    return _wrap


def add_inputs(**kwargs):
    """
    Set metadata associated with a function's input variables.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).add_inputs(**kwargs)
    return _wrap


def add_outputs(**kwargs):
    """
    Set metadata associated with a function's output variables.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).add_outputs(**kwargs)
    return _wrap


def output_names(names):
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
    def _wrap(func):
        kwargs = {n: {} for n in names}
        return _get_fwrapper(func).add_outputs(**kwargs)
    return _wrap


def defaults(**kwargs):
    """
    Update a function's metadata with uniform defaults.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).set_defaults(**kwargs)
    return _wrap


def metadata(**kwargs):
    """
    Update a function's metadata with uniform values.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).set_metadata(**kwargs)
    return _wrap


def declare_option(name, default=None, values=None, types=None, desc='',
                   upper=None, lower=None, check_valid=None, allow_none=False, recordable=True,
                   deprecation=None):
    r"""
    Declare an option.

    Parameters
    ----------
    name : str
        Name of the option.
    default : object or None
        Optional default value that must be valid under the above 3 conditions.
    values : set or list or tuple or None
        Optional list of acceptable option values.
    types : type or tuple of types or None
        Optional type or list of acceptable option types.
    desc : str
        Optional description of the option.
    upper : float or None
        Maximum allowable value.
    lower : float or None
        Minimum allowable value.
    check_valid : function or None
        User-supplied function with arguments (name, value) that raises an exception
        if the value is not valid.
    allow_none : bool
        If True, allow None as a value regardless of values or types.
    recordable : bool
        If True, add to recorder.
    deprecation : str or None
        If None, it is not deprecated. If a str, use as a DeprecationWarning
        during __setitem__ and __getitem__.
    """
    def _wrap(func):
        return _get_fwrapper(func).declare_option(name, default=default, values=values, types=types,
                                                  desc=desc, upper=upper, lower=lower,
                                                  check_valid=check_valid, allow_none=allow_none,
                                                  recordable=recordable, deprecation=deprecation)
    return _wrap


def declare_partials(**kwargs):
    """
    Store declare_partials info in function's metadata.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).declare_partials(**kwargs)
    return _wrap


def declare_coloring(**kwargs):
    """
    Store declare_coloring info in function's metadata.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    def _wrap(func):
        return _get_fwrapper(func).declare_coloring(**kwargs)
    return _wrap


#
# Dev API (for retrieving metadata from the function object)
#


def get_input_meta(func):
    """
    Get an iterator of (name, meta_dict) for each input variable.

    Parameters
    ----------
    func : callable
        Callable object we're retrieving metadata from.
    """
    return _get_fwrapper(func).get_input_meta()


def get_output_meta(func):
    """
    Get an iterator of (name, meta_dict) for each output variable.

    Parameters
    ----------
    func : callable
        Callable object we're retrieving metadata from.
    """
    return _get_fwrapper(func).get_output_meta()


def get_declare_partials(func):
    """
    Get an iterator of (**kwargs) to be passed to each call of declare_partials.

    Parameters
    ----------
    func : callable
        Callable object we're retrieving metadata from.
    """
    return _get_fwrapper(func).get_declare_partials()


def get_declare_coloring(func):
    """
    Get an iterator of (**kwargs) to be passed to each call of declare_coloring.

    Parameters
    ----------
    func : callable
        Callable object we're retrieving metadata from.
    """
    return _get_fwrapper(func).get_declare_colorings()


#
# Implementation details
#

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


class _MetaWrappedFunc(object):
    """
    Storage class for function metadata.

    Metadata is assumed to be added from decorator calls, so ordering is reversed.
    """

    def __init__(self, func):
        self._f = func
        self._defaults = {'val': 1.0, 'shape': ()}
        self._metadata = {}

        # populate _inputs dict with input names based on function signature so we can error
        # check vs. inputs added via add_input
        sig = inspect.signature(func)
        self._inputs = {n: {'val': None if p.default is inspect._empty else p.default} for n, p in sig.parameters.items()}
        self._outputs = {}
        self._declare_partials = []
        self._declare_colorings = []
        self._call_setup = True
        self._use_jax = False

    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def set_defaults(self, **kwargs):
        self._defaults.update(kwargs)
        return self

    def set_metadata(self, **kwargs):
        self._metadata.update(kwargs)
        return self

    def add_input(self, name, **kwargs):
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
        for name, meta in kwargs.items():
            self.add_input(name, meta)
        return self

    def add_output(self, name, **kwargs):
        if name in self._inputs:
            raise RuntimeError(f"'{name}' already registered as an input")
        if name in self._outputs:
            raise RuntimeError(f"'{name}' already registered as an output")
        self._outputs[name] = kwargs
        return self

    def add_outputs(self, *kwargs):
        # because individual add_output calls come from stacked decorators, their order is reversed.
        # The args to add_outputs are in the correct order, so in order to stay consistent
        # with the ordering of add_output, we reverse the order of the args.
        for name, meta in reversed(kwargs.items()):
            self.add_output(name, meta)
        return self

    def declare_option(self, name, **kwargs):
        r"""
        Declare an option.

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
        self._declare_partials.append(kwargs)
        if 'method' in kwargs and kwargs['method'] == 'jax':
            self._use_jax = True
        return self

    def declare_coloring(self, **kwargs):
        self._declare_colorings.append(kwargs)
        return self

    def get_input_meta(self):
        if self._call_setup:
            self._setup()
        return list(self._inputs.items())

    def get_output_meta(self):
        if self._call_setup:
            self._setup()
        return list(reversed(self._outputs.items()))

    def get_declare_partials(self):
        return list(reversed(self._declare_partials.items()))

    def get_declare_colorings(self):
        return list(reversed(self._declare_coloring.items()))

    def _check_vals_equal(self, name, val1, val2):
        # == is more prone to raise exceptions when ndarrays are involved, so use !=
        neq = val1 != val2
        if (isinstance(neq, np.ndarray) and np.any(neq)) or neq:
            raise RuntimeError(f"Conflicting metadata entries for '{name}'.")

    def _resolve_meta(self, name, dct):
        """
        Update the value of the metadata corresponding to name based on defaults, etc.
        """
        if name in self._metadata:
            mval = self._metadata[name]
            if name in dct:
                val = dct[name]
                # check for conflict with func metadata
                if val is None:
                    dct[name] = mval
                else:
                    self._check_vals_equal(name, val, mval)
            else:
                dct[name] = mval

    def _resolve_default(self, name, dct):
        if (name not in dct or dct[name] is None) and name in self._defaults:
            dct[name] = self._defaults[name]

    def _setup(self):
        self._call_setup = False
        overrides = set(self._defaults)
        overrides.update(self._metadata)
        overrides = overrides - {'val', 'shape'}

        self._setup_inputs(overrides)
        self._setup_outputs(overrides)

    def _setup_inputs(self, overrides):
        """
        Populate metadata associated with function inputs.
        """
        ins = self._inputs

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
        outmeta = {}

        # Parse the function code to possibly identify the names of the return values and
        # input/output dependencies. Return names will be non-None only if they are a simple name,
        # e.g., return a, b, c
        outlist = []
        try:
            ret_info = get_function_deps(self._f)
            # if we found return value names by inspection, they're in the correct order, but we
            # reverse them here to be consistent with return value names defined using add_output,
            # which are called as decorators (which execute in inside-out order, which is reversed
            # from what a user would think).
            ret_info.reverse()
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

        full = overrides.union({'val', 'shape'})
        for meta in outs.values():
            for o in full:
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


@contextmanager
def jax_context(globals):
    savenp = savenumpy = None
    if 'np' in globals and globals['np'] is np:
        savenp = globals['np']
    if 'numpy' in globals:
        savenumpy = globals['numpy']
    try:
        yield
    finally:
        if savenp is not None:
            globals['np'] = savenp
        if savenumpy is not None:
            globals['numpy'] = savenumpy


def _get_fwrapper(func):
    """
    Return a wrapped function object.

    If arg is already a wrapped function object, return that.

    Parameters
    ----------
    func : function or _MetaWrappedFunc
        A plain or already wrapped function object.

    Returns
    -------
    _MetaWrappedFunc
        The wrapped function object.
    """
    if isinstance(func, _MetaWrappedFunc):
        return func
    return _MetaWrappedFunc(func)


def _get_long_name(node):
    # If the node is an Attribute or Name node that is composed
    # only of other Attribute or Name nodes, then return the full
    # dotted name for this node. Otherwise, i.e., if this node
    # contains Subscripts or Calls, return None.
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
