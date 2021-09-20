"""
API to associate metadata with and retrieve metadata from function objects.
"""

from numbers import Number
import ast
import inspect
import warnings
import numpy as np

# User API (decorators used to associate metadata with the function)

def in_var(name, **kwargs):
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
        return _get_fwrapper(func).in_var(name, **kwargs)
    return _wrap


def out_var(name, **kwargs):
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
        return _get_fwrapper(func).in_var(name, **kwargs)
    return _wrap


def in_vars(**kwargs):
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
        return _get_fwrapper(func).in_vars(**kwargs)
    return _wrap


def out_vars(**kwargs):
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
        return _get_fwrapper(func).out_vars(**kwargs)
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


# Dev API (for retrieving metadata from the function object)


def get_invar_meta(func):
    return _get_fwrapper(func).get_invar_meta()


def get_outvar_meta(func):
    return _get_fwrapper(func).get_outvar_meta()


def get_declare_partials(func):
    return _get_fwrapper(func).get_declare_partials()


def get_declare_coloring(func):
    return _get_fwrapper(func).get_declare_colorings()


def shape2tuple(shape):
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
        self._defaults = {'val': 1.0}
        self._metadata = {}
        self._invars = {}
        self._outvars = {}
        self._declare_partials = []
        self._declare_colorings = []
        self._call_setup = True

    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def set_defaults(self, **kwargs):
        self._defaults.update(kwargs)
        return self

    def set_metadata(self, **kwargs):
        self._metadata.update(kwargs)
        return self

    def in_var(self, name, **kwargs):
        self._check_already_registered(name)
        self._invars[name] = kwargs
        return self

    def in_vars(self, *kwargs):
        # because individual in_var calls come from stacked decorators, we reverse their
        # order, so we need to reverse what comes in from in_vars in order to stay consistent
        # when retrieving in_var metadata.
        for name, meta in reversed(kwargs.items()):
            self.in_var(name, meta)
        return self

    def get_invar_meta(self):
        if self._call_setup:
            self._setup()
        return list(reversed(self._invars.items()))

    def out_var(self, name, **kwargs):
        self._check_already_registered(name)
        self._outvars[name] = kwargs
        return self

    def out_vars(self, *kwargs):
        # see comment in in_vars about reversal
        for name, meta in reversed(kwargs.items()):
            self.out_var(name, meta)
        return self

    def get_outvar_meta(self):
        if self._call_setup:
            self._setup()
        return list(reversed(self._outvars.items()))

    def declare_partials(self, **kwargs):
        self._declare_partials.append(kwargs)
        return self

    def get_declare_partials(self):
        return list(reversed(self._declare_partials.items()))

    def declare_coloring(self, **kwargs):
        self._declare_colorings.append(kwargs)
        return self

    def get_declare_colorings(self):
        return list(reversed(self._declare_coloring.items()))

    def _check_already_registered(self, name):
        if name in self._invars:
            raise RuntimeError(f"'{name}' already registered as an input")
        if name in self._outvars:
            raise RuntimeError(f"'{name}' already registered as an output")

    def _check_vals_equal(self, name, val1, val2):
        # == is more prone to raise exceptions when ndarrays are involved, so use !=
        neq = val1 != val2
        if (isinstance(neq, np.ndarray) and np.any(neq)) or neq:
            raise RuntimeError(f"Conflicting metadata entries for '{name}'.")

    def _resolve_meta(self, name, dct):
        """
        Update the value of the metadata corresponding to name based on defaults, etc.
        """
        if name in dct:
            val = dct[name]
            # check for conflict with func metadata
            if name in self._metadata:
                mval = self._metadata[name]
                self._check_vals_equal(name, val, mval)
        elif name in self._defaults:
            dct[name] = self._defaults[name]

    def _setup(self):
        self._setup_invars()
        self._setup_outvars()

    def _setup_invars(self):
        """
        Populate metadata associated with function inputs.
        """
        ins = {}  # need to remake the dict to ensure ordering is right

        # first, retrieve inputs from the function signature
        sig = inspect.signature(self._f)
        for name, p in sig.parameters.items():
            ins[name] = meta = {'val': None, 'shape': None}
            if p.default is not inspect._empty:
                meta['val'] = p.default

            if name in self._invars:
                decmeta = self._invars[name]
                if 'val' in decmeta and meta['val'] is not None:
                    self._check_vals_equal(name, decmeta['val'], meta['val'])
                meta.update(decmeta)

            # assume a default value if necessary
            if meta['val'] is None and meta['shape'] is None:
                meta['val'] = self.default_val

            if meta['val'] is not None:
                if np.isscalar(meta['val']):
                    shape = ()
                else:
                    shape = meta['val'].shape

                if meta['shape'] is None:
                    meta['shape'] = shape
                else:
                    meta['shape'] = shape2tuple(meta['shape'])
                    if not shape:  # val is a scalar so reshape with the given meta['shape']
                        meta['val'] = np.ones(meta['shape']) * meta['val']
                    elif shape != meta['shape']:
                        raise ValueError(f"Input '{name}' default value has shape "
                                         f"{shape}, but shape was specified as {meta['shape']}.")

        self._invars = ins

    def _setup_outvars(self):
        outmeta = {}

        # Parse the function code to possibly identify the names of the return values and
        # input/output dependencies. Return names will be non-None only if they are a simple name,
        # e.g., return a, b, c
        outlist = []
        try:
            ret_info = get_function_deps(self._f)
        except RuntimeError:
            #  this could happen if function is compiled or has multiple return lines
            if not self._outvars:
                raise RuntimeError(f"Couldn't determine function return names or "
                                   "number of return values based on AST and no return value "
                                   "annotations were supplied.")
            warnings.warn("Couldn't determine function return names based on AST.  Assuming number "
                          "of return values matches number of return value annotations.")
            outlist = list(self._outvars.items())
        else:
            for o, deps in ret_info:
                outlist.append([o, {'deps': deps}])

        notfound = []
        for oname, ometa in self._outvars.items():
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
                raise RuntimeError(f"{self.msginfo}: Number of unnamed return values "
                                   f"({len(inones)}) doesn't match number of unmatched annotated "
                                   f"return values ({len(notfound)}).")

            # number of None return slots equals number of annotated entries not found in outlist
            for i_olist, name_notfound in zip(inones, notfound):
                annotated_meta = outmeta[name_notfound]
                _, ret_meta = outlist[i_olist]
                ret_meta.update(annotated_meta)
                outlist[i_olist] = (name_notfound, ret_meta)

        outs = {n: m for n, m in outlist}

        self._compute_out_shapes(func, ins, outs, use_jax)

        self._outvars = outs

    def _compute_out_shapes(self, func, ins, outs, use_jax=True):
        """
        Compute the shapes of outputs based on those of the inputs.

        Parameters
        ----------
        func : function
            The function whose outputs' shapes will be determined.
        ins : dict
            Dict of input metadata containing input shapes.
        outs : dict
            Dict of output metadata that will be updated with shape information.
        """
        need_shape = False
        for ometa in outs.values():
            try:
                ometa['shape']
            except KeyError:
                need_shape = True
                break

        args = []
        for name, meta in ins.items():
            if meta['val'] is not None:
                args.append(meta['val'])
            else:
                try:
                    shp = meta['shape']
                except KeyError:
                    raise RuntimeError(f"{self.msginfo}: Can't determine shape of input '{name}'.")
                if jax is not None:
                    args.append(jax.ShapedArray(shape2tuple(shp), dtype=np.float64))

        if need_shape:  # output shapes weren't provided by annotations
            if jax is None:
                raise RuntimeError(f"{self.msginfo}: Some return values have unknown shape. Jax "
                                   "can (possibly) determine the output shapes based on the input "
                                   "shapes, but jax was not found.  Either install jax (pip "
                                   "install jax), or add return value annotations to the function "
                                   "that specify the shapes of return values.")

        if jax is not None:  # compute shapes as a check against annotated value (if any)
            # must replace numpy with jax numpy when making jaxpr.
            if 'np' in func.__globals__:
                func.__globals__['np'] = jnp
            try:
                v = jax.make_jaxpr(func)(*args)
            except Exception as err:
                if need_shape:
                    raise RuntimeError(f"{self.msginfo}: Failed to determine the output shapes "
                                       f"based on the input shapes. The error was: {err}.  To "
                                       "avoid this error, add return value annotations that "
                                       "specify the shapes of the return values to the function.")
                if use_jax:
                    issue_warning("Failed to determine the output shapes based on the input "
                                  "shapes in order to check the provided annotated values.  The "
                                  f"error was: {err}.", prefix=self.msginfo)
            else:
                for val, name in zip(v.out_avals, outs):
                    oldshape = outs[name].get('shape')
                    if oldshape is not None and oldshape != val.shape:
                        raise RuntimeError(f"{self.msginfo}: Annotated shape for return value "
                                           f"'{name}' of {oldshape} doesn't match computed shape "
                                           f"of {val.shape}.")
                    outs[name]['shape'] = val.shape
            finally:
                if 'np' in func.__globals__:
                    func.__globals__['np'] = np


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
            raise RuntimeError("_ReturnNamesCollector does not support multiple returns in a "
                               "single function.  Either the given function contains multiple "
                               "returns or this _ReturnNamesCollector instance has been used "
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
