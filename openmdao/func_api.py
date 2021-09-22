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


#
# User API (decorators used to associate metadata with the function)
#

def add_input(name, val=None, shape=None, src_indices=None, flat_src_indices=None,
              units=None, desc=None, tags=None, shape_by_conn=None, copy_shape=None,
              distributed=None, new_style_idx=None):
    """
    Set metadata associated with one of a function's input variables.

    Parameters
    ----------
    name : str
        Name of the variable in this component's namespace.
    val : float or list or tuple or ndarray or Iterable
        The initial value of the variable being added in user-defined units.
    shape : int or tuple or list or None
        Shape of this variable, only required if src_indices not provided and
        val is not an array. Default is None.
    src_indices : int or list or tuple or int ndarray or Iterable or None
        The global indices of the source variable to transfer data from.
        A value of None implies this input depends on all entries of the source array.
        Default is None. The shapes of the target and src_indices must match,
        and the form of the entries within is determined by the value of 'flat_src_indices'.
    flat_src_indices : bool
        If True and the source is non-flat, each entry of src_indices is assumed to be an index
        into the flattened source.  Ignored if the source is flat.
    units : str or None
        Units in which this input variable will be provided to the component
        during execution. Default is None, which means it is unitless.
    desc : str
        Description of the variable.
    tags : str or list of strs
        User defined tags that can be used to filter what gets listed when calling
        list_inputs and list_outputs.
    shape_by_conn : bool
        If True, shape this input to match its connected output.
    copy_shape : str or None
        If a str, that str is the name of a variable. Shape this input to match that of
        the named variable.
    distributed : bool
        If True, this variable is a distributed variable, so it can have different sizes/values
        across MPI processes.
    new_style_idx : bool
        If True, assume numpy compatible indexing.  Not setting this to True will result in a
        deprecation warning for src_indices arrays with ndim > 1.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    loc = locals()
    def _wrap(func):
        return _get_fwrapper(func).add_input(**_get_kwargs(add_input, loc))
    return _wrap


def add_output(name, val=None, shape=None, units=None, res_units=None, desc=None,
               lower=None, upper=None, ref=None, ref0=None, res_ref=None, tags=None,
               shape_by_conn=False, copy_shape=None, distributed=None):
    """
    Set metadata associated with one of a function's return values.

    Parameters
    ----------
    name : str
        Name of the variable in this component's namespace.
    val : float or list or tuple or ndarray
        The initial value of the variable being added in user-defined units. Default is 1.0.
    shape : int or tuple or list or None
        Shape of this variable, only required if val is not an array.
        Default is None.
    units : str or None
        Units in which the output variables will be provided to the component during execution.
        Default is None, which means it has no units.
    res_units : str or None
        Units in which the residuals of this output will be given to the user when requested.
        Default is None, which means it has no units.
    desc : str
        Description of the variable.
    lower : float or list or tuple or ndarray or Iterable or None
        Lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
        consistent with the shape arg (if given), or (3) an array_like matching the shape of
        val, if val is array_like. A value of None means this output has no lower bound.
        Default is None.
    upper : float or list or tuple or ndarray or or Iterable None
        Upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
        consistent with the shape arg (if given), or (3) an array_like matching the shape of
        val, if val is array_like. A value of None means this output has no upper bound.
        Default is None.
    ref : float or ndarray
        Scaling parameter. The value in the user-defined units of this output variable when
        the scaled value is 1. Default is 1.
    ref0 : float or ndarray
        Scaling parameter. The value in the user-defined units of this output variable when
        the scaled value is 0. Default is 0.
    res_ref : float or ndarray
        Scaling parameter. The value in the user-defined res_units of this output's residual
        when the scaled value is 1. Default is 1.
    tags : str or list of strs or set of strs
        User defined tags that can be used to filter what gets listed when calling
        list_inputs and list_outputs.
    shape_by_conn : bool
        If True, shape this output to match its connected input(s).
    copy_shape : str or None
        If a str, that str is the name of a variable. Shape this output to match that of
        the named variable.
    distributed : bool
        If True, this variable is a distributed variable, so it can have different sizes/values
        across MPI processes.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    loc = locals()
    def _wrap(func):
        return _get_fwrapper(func).add_output(**_get_kwargs(add_output, loc))
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
    _check_kwargs(kwargs, _allowed_add_input_args)
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
    _check_kwargs(kwargs, _allowed_add_output_args)
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


def declare_option(name, default=None, values=None, types=None, desc=None,
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
    loc = locals()
    def _wrap(func):
        return _get_fwrapper(func).declare_option(**_get_kwargs(declare_option, loc))
    return _wrap


def declare_partials(of, wrt, dependent=True, rows=None, cols=None, val=None,
                     method='exact', step=None, form=None, step_calc=None, minimum_step=None):
    """
    Store declare_partials info in function's metadata.

    Parameters
    ----------
    of : str or list of str
        The name of the residual(s) that derivatives are being computed for.
        May also contain a glob pattern.
    wrt : str or list of str
        The name of the variables that derivatives are taken with respect to.
        This can contain the name of any input or output variable.
        May also contain a glob pattern.
    rows : ndarray of int or None
        Row indices for each nonzero entry.  For sparse subjacobians only.
    cols : ndarray of int or None
        Column indices for each nonzero entry.  For sparse subjacobians only.
    val : float or ndarray of float or scipy.sparse
        Value of subjacobian.  If rows and cols are not None, this will
        contain the values found at each (row, col) location in the subjac.
    method : str
        The type of approximation that should be used. Valid options include:
        'fd': Finite Difference, 'cs': Complex Step, 'exact': use the component
        defined analytic derivatives. Default is 'exact'.
    step : float
        Step size for approximation. Defaults to None, in which case the approximation
        method provides its default value.
    form : str
        Form for finite difference, can be 'forward', 'backward', or 'central'. Defaults
        to None, in which case the approximation method provides its default value.
    step_calc : str
        Step type for computing the size of the finite difference step. It can be 'abs' for
        absolute, 'rel_avg' for a size relative to the absolute value of the vector input, or
        'rel_element' for a size relative to each value in the vector input. In addition, it
        can be 'rel_legacy' for a size relative to the norm of the vector.  For backwards
        compatibilty, it can be 'rel', which currently defaults to 'rel_legacy', but in the
        future will default to 'rel_avg'. Defaults to None, in which case the approximation
        method provides its default value.
    minimum_step : float
        Minimum step size allowed when using one of the relative step_calc options.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    loc = locals()
    def _wrap(func):
        return _get_fwrapper(func).declare_partials(**_get_kwargs(declare_partials, loc))
    return _wrap


def declare_coloring(wrt=None, method=None, form=None, step=None, per_instance=None,
                     num_full_jacs=None, tol=None, orders=None, perturb_size=None,
                     min_improve_pct=None, show_summary=None, show_sparsity=None):
    """
    Store declare_coloring info in function's metadata.

    Parameters
    ----------
    wrt : str or list of str
        The name or names of the variables that derivatives are taken with respect to.
        This can contain input names, output names, or glob patterns.
    method : str
        Method used to compute derivative: "fd" for finite difference, "cs" for complex step.
    form : str
        Finite difference form, can be "forward", "central", or "backward". Leave
        undeclared to keep unchanged from previous or default value.
    step : float
        Step size for finite difference. Leave undeclared to keep unchanged from previous
        or default value.
    per_instance : bool
        If True, a separate coloring will be generated for each instance of a given class.
        Otherwise, only one coloring for a given class will be generated and all instances
        of that class will use it.
    num_full_jacs : int
        Number of times to repeat partial jacobian computation when computing sparsity.
    tol : float
        Tolerance used to determine if an array entry is nonzero during sparsity determination.
    orders : int
        Number of orders above and below the tolerance to check during the tolerance sweep.
    perturb_size : float
        Size of input/output perturbation during generation of sparsity.
    min_improve_pct : float
        If coloring does not improve (decrease) the number of solves more than the given
        percentage, coloring will not be used.
    show_summary : bool
        If True, display summary information after generating coloring.
    show_sparsity : bool
        If True, display sparsity with coloring info after generating coloring.

    Returns
    -------
    function
        A function wrapper that updates the function's metadata.
    """
    loc = locals()
    def _wrap(func):
        return _get_fwrapper(func).declare_coloring(**_get_kwargs(declare_coloring, loc))
    return _wrap


def apply_decorators(func, *decorators):
    r"""
    Apply the given list of decorators to the given function.

    Parameters
    ----------
    func : function
        Function to be decorated.
    *decorators : list
        List of decorators to apply to the given function.

    Returns
    -------
    callable
        Decorated function.
    """
    for dec in decorators:
        func = dec(func)
    return func

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

    Returns
    -------
    iter of (name, dict)
        An iterator of (input_name, input_meta) tuples.
    """
    return _get_fwrapper(func).get_input_meta()


def get_output_meta(func):
    """
    Get an iterator of (name, meta_dict) for each output variable.

    Parameters
    ----------
    func : callable
        Callable object we're retrieving metadata from.

    Returns
    -------
    iter of (name, dict)
        An iterator of (output_name, output_meta) tuples.
    """
    return _get_fwrapper(func).get_output_meta()


def get_declare_partials(func):
    """
    Get an iterator of (**kwargs) to be passed to each call of declare_partials.

    Parameters
    ----------
    func : callable
        Callable object we're retrieving metadata from.

    Returns
    -------
    iter of dict
        An iterator of keyword args for each declare_partials call.
    """
    return _get_fwrapper(func).get_declare_partials()


def get_declare_colorings(func):
    """
    Get an iterator of (**kwargs) to be passed to each call of declare_coloring.

    Parameters
    ----------
    func : callable
        Callable object we're retrieving metadata from.

    Returns
    -------
    iter of dict
        An iterator of keyword args for each declare_coloring call.
    """
    return _get_fwrapper(func).get_declare_colorings()


#
# Implementation details
#

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


class _MetaWrappedFunc(object):
    """
    Storage class for function metadata.

    Metadata is assumed to be added from decorator calls, so ordering is reversed.

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

    def set_defaults(self, **kwargs):
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

    def set_metadata(self, **kwargs):
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
        # because individual add_output calls come from stacked decorators, their order is reversed.
        # The args to add_outputs are in the correct order, so in order to stay consistent
        # with the ordering of add_output, we reverse the order of the args.
        for name, meta in reversed(kwargs.items()):
            self.add_output(name, meta)
        return self

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
        return reversed(self._outputs.items())

    def get_declare_partials(self):
        """
        Get an iterator of keyword args passed to each declare_partials decorator call.

        Returns
        -------
        iter of dict
            Iterator of dicts containing the keyword args for each call in top to bottom order.
        """
        return reversed(self._declare_partials.items())

    def get_declare_colorings(self):
        """
        Get an iterator of keyword args passed to each declare_coloring decorator call.

        Returns
        -------
        iter of dict
            Iterator of dicts containing the keyword args for each call in top to bottom order.
        """
        return reversed(self._declare_coloring.items())

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
