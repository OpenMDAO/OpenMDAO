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


_allowed_add_input_args = {
    'val', 'shape', 'units', 'desc', 'tags', 'shape_by_conn', 'copy_shape', 'distributed',
    'new_style_idx'
}

_allowed_add_output_args = {
    'val', 'shape', 'units', 'res_units', 'desc' 'lower', 'upper', 'ref', 'ref0', 'res_ref', 'tags',
    'shape_by_conn', 'copy_shape', 'distributed', 'resid'
}

_allowed_declare_options_args = {
    'default', 'values', 'types', 'desc', 'upper', 'lower', 'check_valid', 'allow_none',
    'recordable', 'deprecation'
}

_allowed_declare_partials_args = {
    'of', 'wrt', 'dependent', 'rows', 'cols', 'val', 'method', 'step', 'form', 'step_calc',
    'minimum_step'
}

_allowed_declare_coloring_args = {
    'wrt', 'method', 'form', 'step', 'per_instance', 'num_full_jacs', 'tol', 'orders',
    'perturb_size', 'min_improve_pct', 'show_summary', 'show_sparsity'
}


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
    _inputs : dict
        Dict of metadata dicts keyed to input name.
    _outputs : dict
        Dict of metadata dicts keyed to output name.
    _declare_partials : list
        List of keyword args, one entry for each call to declare_partials.
    _declare_coloring : dict
        Keyword args for call to declare_coloring.
    _call_setup : bool
        If True, call the setup functions for input and output metadata.
    _use_jax : bool
        If True, use jax to compute output shapes based on input shapes.
    """

    def __init__(self, func):
        """
        Initialize attributes.
        """
        self._f = func
        self._input_defaults = {'val': 1.0, 'shape': ()}
        self._output_defaults = {'val': 1.0, 'shape': ()}
        self._partials_defaults = {}
        self._coloring_defaults = {}

        # populate _inputs dict with input names based on function signature so we can error
        # check vs. inputs added via add_input
        self._inputs = {n: {'val': None if p.default is inspect._empty else p.default}
                        for n, p in inspect.signature(func).parameters.items()}
        self._outputs = {}
        self._declare_partials = []
        self._declare_coloring = None
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
        Add default metadata that may apply to the wrapped function.

        Any variable specific or partials/coloring specific metadata will override any metadata
        specified here.

        Parameters
        ----------
        **kwargs : dict
            Metadata names and their values.
        """
        input_kwargs = _filter_dict(kwargs, _allowed_add_input_args)
        output_kwargs = _filter_dict(kwargs, _allowed_add_output_args)
        partials_kwargs = _filter_dict(kwargs, _allowed_declare_partials_args)
        coloring_kwargs = _filter_dict(kwargs, _allowed_declare_coloring_args)

        combined = (set(input_kwargs)
                    .union(output_kwargs)
                    .union(partials_kwargs)
                    .union(coloring_kwargs))

        if len(kwargs) > len(combined):
            invalids = (set(kwargs) - _allowed_add_input_args - _allowed_add_output_args -
                        _allowed_declare_partials_args - _allowed_declare_coloring_args)
            raise NameError(f"In defaults, metadata names {sorted(invalids)} are not allowed.")

        self._input_defaults.update(input_kwargs)
        self._output_defaults.update(output_kwargs)
        self._partials_defaults.update(partials_kwargs)
        self._coloring_defaults.update(coloring_kwargs)

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
            raise NameError(f"In add_input, '{name}' is not an input to this function.")
        meta = self._inputs[name]
        for kw in kwargs:
            if kw in meta and meta[kw] is not None:
                raise RuntimeError(f"In add_input, metadata '{kw}' has already been added to "
                                   f"function for input '{name}'.")
        if meta.get('val') is not None and kwargs.get('val') is not None:
            self._check_vals_equal(name, meta['val'], kwargs['val'])
        _check_kwargs(kwargs, _allowed_add_input_args, 'add_input')
        meta.update(kwargs)
        return self

    def add_inputs(self, **kwargs):
        r"""
        Add metadata for multiple inputs of the wrapped function.

        Parameters
        ----------
        **kwargs : dict
            Keyword args to store.  The value corresponding to each key is a dict containing the
            metadata for the input name that matches that key.
        """
        for name, meta in kwargs.items():
            self.add_input(name, **meta)
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
            if 'resid' in kwargs:
                self._inputs[name]['resid'] = kwargs['resid']
            else:
                raise RuntimeError(f"In add_output, '{name}' already registered as an input.")
        if name in self._outputs:
            raise RuntimeError(f"In add_output, '{name}' already registered as an output.")
        _check_kwargs(kwargs, _allowed_add_output_args, 'add_output')
        self._outputs[name] = kwargs
        return self

    def add_outputs(self, **kwargs):
        r"""
        Add metadata for multiple outputs of the wrapped function.

        Parameters
        ----------
        **kwargs : dict
            Keyword args to store.  The value corresponding to each key is a dict containing the
            metadata for the output name that matches that key.
        """
        for name, meta in kwargs.items():
            self.add_output(name, **meta)
        return self

    def output_names(self, *names):
        r"""
        Set the names of a function's output variables.

        Parameters
        ----------
        *names : list of str
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
        _check_kwargs(kwargs, _allowed_declare_options_args, 'declare_option')
        self._inputs[name].update(kwargs)
        self._inputs[name]['is_option'] = True
        return self

    def declare_partials(self, of=('*',), wrt=('*',), **kwargs):
        r"""
        Collect args to be passed to declare_partials on an OpenMDAO component.

        Parameters
        ----------
        of : str or list of str
            Individual name/glob pattern or list of names/glob patterns to match
            'of' variables.
        wrt : str or list of str
            Individual name/glob pattern or list of names/glob patterns to match
            'with respect to' variables.
        **kwargs : dict
            Keyword args to store.
        """
        _check_kwargs(kwargs, _allowed_declare_partials_args, 'declare_partials')
        _update_from_defaults(kwargs, self._partials_defaults)

        jaxerr = False
        if 'method' in kwargs and kwargs['method'] == 'jax':
            if self._declare_partials and not self._use_jax:
                jaxerr = True
            self._use_jax = True
        elif self._use_jax:
            jaxerr = True

        if jaxerr:
            raise RuntimeError("If multiple calls to declare_partials() are made on the same "
                               "function object and any set method='jax', then all must set "
                               "method='jax'.")

        kwargs = kwargs.copy()
        kwargs['of'] = of
        kwargs['wrt'] = wrt
        self._declare_partials.append(kwargs)

        return self

    def declare_coloring(self, **kwargs):
        r"""
        Collect args to be passed to declare_coloring on an OpenMDAO component.

        Parameters
        ----------
        **kwargs : dict
            Keyword args to store.
        """
        if self._declare_coloring is None:
            _check_kwargs(kwargs, _allowed_declare_coloring_args, 'declare_coloring')
            _update_from_defaults(kwargs, self._coloring_defaults)
            self._declare_coloring = kwargs.copy()
            return self
        raise RuntimeError("declare_coloring has already been called.")

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
        return [it for it in self._inputs.items() if 'resid' not in it[1]]

    def get_input_names(self):
        """
        Get an iterator over input variable names.

        Yields
        ------
        str
            Name of each input variable.
        """
        for name, _ in self.get_input_meta():
            yield name

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

    def get_output_names(self):
        """
        Get an iterator over output variable names.

        Yields
        ------
        str
            Name of each output variable.
        """
        for name, _ in self.get_output_meta():
            yield name

    def get_declare_partials(self):
        """
        Get an iterator of keyword args passed to each declare_partials call.

        Returns
        -------
        iter of dict
            Iterator of dicts containing the keyword args for each call.
        """
        return self._declare_partials

    def get_declare_coloring(self):
        """
        Get keyword args passed to declare_coloring call.

        Returns
        -------
        iter of dict
            Iterator of dicts containing the keyword args for each call.
        """
        return self._declare_coloring

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

    def _setup(self):
        """
        Set up input and output variable metadata dicts.
        """
        self._call_setup = False

        self._setup_inputs()
        self._setup_outputs()

    def _setup_inputs(self):
        """
        Set up the input variable metadata dicts.
        """
        ins = self._inputs
        outs = self._outputs

        # first, retrieve inputs from the function signature
        for name in inspect.signature(self._f).parameters:
            if name in outs:  # skip if this is a state
                continue

            meta = ins[name]

            if meta.get('is_option'):
                continue

            self._default_to_shape(name, meta, self._input_defaults)
            _update_from_defaults(meta, self._input_defaults)

    def _setup_outputs(self):
        """
        Set up the output variable metadata dicts.
        """
        # Parse the function code to possibly identify the names of the return values Return names
        # will be non-None only if they are a simple name, e.g., return a, b, c
        outlist = []
        try:
            outlist = [(n, {}) for n in self.get_return_names()]
        except RuntimeError:
            #  this could happen if function is compiled or has multiple return lines
            if not self._outputs:
                raise RuntimeError(f"Couldn't determine function return names or "
                                   "number of return values based on AST and no return value "
                                   "metadata was supplied.")
            warnings.warn("Couldn't determine function return names based on AST.  Assuming number "
                          "of return values matches number of outputs defined in the metadata.")
            outlist = list(self._outputs.items())

        residmap = {n: n for n, _ in outlist}
        notfound = []
        for oname, ometa in self._outputs.items():
            residmap[oname] = oname
            for n, meta in outlist:
                if n == oname or 'resid' in ometa and ometa['resid'] == n:
                    if meta is not ometa:
                        meta.update(ometa)
                    residmap[n] = oname
                    break
            else:  # didn't find oname
                notfound.append(oname)

        if notfound:  # try to fill in the unnamed slots with user-supplied output data
            inones = [i for i, t in enumerate(outlist) if t[0] is None]  # indices with no name
            if len(notfound) != len(inones):
                raise RuntimeError(f"There must be an unnamed return value for every unmatched "
                                   f"output name {notfound} but only found {len(inones)}.")

            # number of None return slots equals number of entries not found in outlist
            for i_olist, name_notfound in zip(inones, notfound):
                m = self._outputs[name_notfound]
                _, ret_meta = outlist[i_olist]
                ret_meta.update(m)
                outlist[i_olist] = (name_notfound, ret_meta)

        outs = {residmap[n]: m for n, m in outlist}

        if self._use_jax:
            # make sure jax used for all declared derivs
            self._compute_out_shapes(self._inputs, outs)

        for name, meta in outs.items():
            self._default_to_shape(name, meta, self._output_defaults)
            _update_from_defaults(meta, self._output_defaults)
            if meta['shape'] is not None:
                meta['shape'] = _shape2tuple(meta['shape'])

        self._outputs = outs

    def get_return_names(self):
        """
        Return list of return value names.

        Returns
        -------
        list
            List of names containing one entry for each return value.  Each name will be the
            name of the return value if it has a simple name, otherwise None.
        """
        input_names = set(self.get_input_names())
        return [n if n not in input_names else None
                for n in _FuncRetNameCollector(self._f)._ret_info]

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
        static_argnums = []
        for i, (name, meta) in enumerate(ins.items()):
            if 'is_option' in meta and meta['is_option']:
                if 'default' in meta:
                    val = meta['default']
                elif 'values' in meta:
                    val = meta['values'][0]
                else:
                    val = None
                args.append(val)
                static_argnums.append(i)
                continue
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
        if jax is not None:
            try:
                # must replace numpy with jax numpy when making jaxpr.
                with jax_context(self._f.__globals__):
                    v = jax.make_jaxpr(self._f, static_argnums)(*args)
            except Exception as err:
                if need_shape:
                    raise RuntimeError(f"Failed to determine the output shapes "
                                       f"based on the input shapes. The error was: {err}.  To "
                                       "avoid this error, add return value metadata that "
                                       "specify the shapes of the return values to the function.")
                warnings.warn("Failed to determine the output shapes based on the input "
                              "shapes in order to check the provided metadata values. The"
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
            shape = self._output_defaults['shape']
            warnings.warn(f"Return values {need_shape} have unspecified shape so are assumed to "
                          f"have shape {shape}.")
            for name in need_shape:
                outs[name]['shape'] = shape

    def _default_to_shape(self, name, meta, defaults_dict):
        if 'val' in meta and meta['val'] is not None:
            valshape = np.asarray(meta['val']).shape
        else:
            valshape = None
            meta['val'] = defaults_dict['val']

        if meta.get('shape') is None:
            if valshape is not None:
                meta['shape'] = valshape
            else:
                meta['shape'] = defaults_dict['shape']

        meta['shape'] = _shape2tuple(meta['shape'])
        if not valshape:  # val is a scalar so reshape with the given meta['shape']
            meta['val'] = np.ones(meta['shape']) * meta['val']
        elif valshape != meta['shape']:
            raise ValueError(f"Input '{name}' default value has shape "
                             f"{valshape}, but shape was specified as {meta['shape']}.")


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


def _update_from_defaults(meta, defaults):
    """
    Update values of the metadata corresponding to defaults.

    Parameters
    ----------
    meta : dict
        The metadata dict to be updated.
    defaults : dict
        The defaults dict.
    """
    for key, val in defaults.items():
        if key not in meta or meta[key] is None:
            meta[key] = val


def _filter_dict(kwargs, allowed):
    return {k: v for k, v in kwargs.items() if k in allowed}


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
        raise RuntimeError(f"In {fname}, metadata names {errs} are not allowed.")


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
    Create a context where np and numpy are replaced by their jax equivalents.

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


class _FuncRetNameCollector(ast.NodeVisitor):
    """
    An ast.NodeVisitor that records return value names.

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
        self._ret_info = []
        self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec'))

    def _get_return_attrs(self, node):
        if isinstance(node, ast.Name):
            self._ret_info.append(node.id)
        else:
            self._ret_info.append(None)

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
