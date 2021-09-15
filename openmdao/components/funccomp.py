"""Define the FuncComponent class."""

try:
    import jax
    from jax import jvp, vjp, vmap, random, jit
    import jax.numpy as jnp
except ImportError:
    jax = None

import re
import inspect
import numpy as np
from numpy import asarray, isscalar, ndarray, imag, complex as npcomplex
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
from openmdao.utils.units import valid_units
from openmdao.utils.code_utils import get_function_deps
from openmdao.utils.general_utils import shape2tuple
from openmdao.utils.om_warnings import issue_warning


# regex to check for variable names.
namecheck_rgx = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

# Names that are not allowed for input or output variables (keywords for options)
_disallowed_varnames = {'has_diag_partials', 'units', 'shape', 'shape_by_conn', 'run_root_only'}


_allowed_add_input_args = {
    'val', 'shape', 'src_indices', 'flat_src_indices', 'units', 'desc', 'tags', 'shape_by_conn',
    'copy_shape', 'distributed', 'new_style_idx',
}

_allowed_add_output_args = {
    'val', 'shape', 'units', 'res_units', 'desc' 'lower', 'upper', 'ref', 'ref0', 'res_ref', 'tags',
    'shape_by_conn', 'copy_shape', 'distributed',
}

_meta_keep = {'units', 'shape', 'val'}
_from_def = {'default_units': 'units', 'default_shape': 'shape'}


def _check_units_option(option, value):
    if value is not None and not valid_units(value):
        raise ValueError(f"The units '{value}' are invalid.")


def _copy_with_ignore(dct, keepers, warn=True):
    """
    Copy the entries in the given dict whose keys are in keepers.

    Parameters
    ----------
    dct : dict
        The dictionary to be copied.
    keepers : set-like
        Set of keys for entries we want to keep.
    warn : bool
        If True, issue a warning showing which keys were ignored.

    Returns
    -------
    dict
        A new dict containing 'keepers' entries.
    set
        The set of ignored keys.
    """
    kept = {}
    ignored = set()
    for k, v in dct.items():
        if k in keepers:
            kept[k] = v
        else:
            ignored.add(k)

    if warn and ignored:
        issue_warning(f"The following metadata entries were ignored: {sorted(ignored)}.")

    return kept, ignored


class ExplicitFuncComp(ExplicitComponent):
    """
    A component that wraps a python function.

    Parameters
    ----------
    func : function
        The function to be wrapped by this Component.
    **kwargs : named args
        Args passed down to ExplicitComponent.

    Attributes
    ----------
    _func : function
        The function wrapped by this component.
    _inmeta : dict
        Function input metadata.
    _outmeta : dict
        Function return value metadata.
    complex_stepsize : float
        Step size used for complex step.
    """

    def __init__(self, func, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._func = func
        self._inmeta = None
        self._outmeta = None
        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-40

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('has_diag_partials', types=bool, default=False,
                             desc='If True, treat all array/array partials as diagonal if both '
                                  'arrays have size > 1. All arrays with size > 1 must have the '
                                  'same flattened size or an exception will be raised.')

        self.options.declare('default_units', types=str, allow_none=True, default=None,
                             desc='Default units to assign to variables in this component. '
                                  'Any units provided for individual variables will override this.',
                             check_valid=_check_units_option)

        self.options.declare('default_shape', types=(int, tuple, list), allow_none=True,
                             default=None,
                             desc='Default shape to assign to variables in this component. '
                                  'Any shape provided for individual variables will override this.')

        self.options.declare('use_jax', types=bool, default=False,
                             desc='If True, use jax to compute derivatives.')

        # self.options.declare('shape_by_conn', types=bool, default=False,
        #                      desc='If True, shape all inputs and outputs based on their '
        #                           'connection. Default is False.')

    def setup(self):
        """
        Define out inputs and outputs.
        """
        self._inmeta, self._outmeta = self._get_func_info(self._func, self.options)

        for name, meta in self._inmeta.items():
            self._check_var_name(name)
            kwargs, _ = _copy_with_ignore(meta, _allowed_add_input_args, warn=True)
            if kwargs['val'] is None:
                kwargs['val'] = 1.0
            self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._outmeta.items()):
            if name is None:
                raise RuntimeError(f"{self.msginfo}: Can't add output corresponding to return "
                                   f"value in position {i} because it has no name.  Specify the "
                                   "name by returning a variable, for example 'return myvar', or "
                                   "include the name in the function's return value annotation.")
            self._check_var_name(name)
            kwargs, ignored = _copy_with_ignore(meta, _allowed_add_output_args, warn=False)
            ignored.remove('deps')
            if ignored:
                issue_warning(f"The following metadata entries were ignored: {sorted(ignored)}.")
            self.add_output(name, **kwargs)

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        meta = self._var_rel2meta
        decl_partials = super().declare_partials
        hasdiag = self.options['has_diag_partials']
        for i, (out, ometa) in enumerate(sorted(self._outmeta.items(), key=lambda x: x[0])):
            oshp = ometa['shape']
            if not oshp:
                osize = 1
            else:
                osize = np.product(oshp) if isinstance(oshp, tuple) else oshp

            inds = np.arange(osize, dtype=INT_DTYPE)
            for inp, imeta in self._inmeta.items():
                if inp not in ometa['deps']:
                    continue

                if hasdiag:
                    ishp = imeta['shape']
                    if not ishp:
                        isize = 1
                    else:
                        isize = np.product(ishp) if isinstance(oshp, tuple) else ishp
                    if osize != isize:
                        raise RuntimeError(f"{self.msginfo}: has_diag_partials is True but "
                                           f"partial({out}, {inp}) is not square "
                                           f"(shape=({osize}, {isize})).")
                    # partial will be declared as diagonal
                    if osize > 1:
                        decl_partials(of=out, wrt=inp, rows=inds, cols=inds)
                    else:
                        decl_partials(of=out, wrt=inp)
                else:
                    decl_partials(of=out, wrt=inp)

        super()._setup_partials()

    def _check_var_name(self, name):
        match = namecheck_rgx.match(name)
        if match is None or match.group() != name:
            raise NameError(f"{self.msginfo}: '{name}' is not a valid variable name." )

        if name in _disallowed_varnames:
            raise NameError(f"{self.msginfo}: cannot use variable name '{name}' because "
                            "it's a reserved keyword.")

    def _compute_output_array(self, input_values, output_array):
        """
        Fill the given output array with our function result based on the given input values.

        Parameters
        ----------
        input_values : tuple of ndarrays or floats
            Unscaled, dimensional input variables.
        output_array
            The output array being filled.
        """
        outs = self._func(*input_values)
        if isinstance(outs, tuple):
            start = end = 0
            for o in outs:
                a = asarray(o) if isscalar(o) else o
                end += a.size
                output_array[start:end] = a.flat
                start = end
        else:
            if isscalar(outs):
                output_array[:] = outs
            else:
                output_array[:] = outs.flat

    def compute(self, inputs, outputs):
        """
        Compute the result of calling our function with the given inputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables.
        outputs : Vector
            Unscaled, dimensional output variables.
        """
        # this will update the outputs array in place
        self._compute_output_array(inputs.values(), outputs.asarray())

    def compute_partials(self, inputs, partials):
        """
        Use complex step method to update the given Jacobian.

        Parameters
        ----------
        inputs : `VecWrapper`
            `VecWrapper` containing parameters (p).
        partials : `Jacobian`
            Contains sub-jacobians.
        """
        step = self.complex_stepsize * 1j
        inv_stepsize = 1.0 / self.complex_stepsize
        has_diag_partials = self.options['has_diag_partials']

        in_vals = [np.asarray(v, dtype=npcomplex) for v in inputs.values()]
        result = np.zeros(len(self._outputs), dtype=npcomplex)
        out_slices = self._outputs.get_slice_dict()

        for ivar, inp in enumerate(inputs._abs_iter()):

            if has_diag_partials or in_vals[ivar].size == 1:
                # set a complex input value
                in_vals[ivar] += step

                # solve with complex input value
                self._compute_output_array(in_vals, result)

                for u, slc in out_slices.items():
                    if (u, inp) in self._subjacs_info:
                        partials[(u, inp)] = imag(result[slc] * inv_stepsize)

                # restore old input value
                in_vals[ivar] -= step
            else:
                pval = in_vals[ivar]
                if np.isscalar(pval):
                    # set a complex input value
                    in_vals[ivar] += step

                    # solve with complex input value
                    self._compute_output_array(in_vals, result)

                    for u, slc in out_slices.items():
                        if (u, inp) in self._subjacs_info:
                            # set the column in the Jacobian entry
                            partials[(u, inp)][:, i] = imag(result[slc] * inv_stepsize)

                    # restore old input value
                    in_vals[ivar] -= step
                else:
                    for i in range(pval.size):
                        # set a complex input value
                        in_vals[ivar].flat[i] += step

                        # solve with complex input value
                        self._compute_output_array(in_vals, result)

                        for u, slc in out_slices.items():
                            if (u, inp) in self._subjacs_info:
                                # set the column in the Jacobian entry
                                partials[(u, inp)][:, i] = imag(result[slc] * inv_stepsize)

                        # restore old input value
                        in_vals[ivar].flat[i] -= step

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


    def _get_func_info(self, func, compmeta=None, default_val=1.0):
        """
        Retrieve metadata associated with function inputs and return values.

        Return value metadata can come from annotations or (shape only) can be determined
        using jax if the input shapes or values are known.  Return value names can be defined
        in annotations or can be determined from the function itself provided that the return
        values are internal function variable names.

        Parameters
        ----------
        func : function
            The function to be queried for input and return value info.
        compmeta : dict or None
            Dict containing component wide options like shape and units.
        default_val : object
            If True, set the default value of any input without one to this value.

        Returns
        -------
        dict
            Dictionary of metdata for inputs.
        dict
            Dictionary of metadata for return values.
        """
        ins = {}

        if compmeta is None:
            compmeta = {}
        else:
            # convert default_* names
            compmeta = {_from_def.get(k, k): v for k, v in compmeta.items()}

        funcmetalist = func.__annotations__.get(':meta', {}).get('func_meta')
        funcmeta = {}
        if funcmetalist:
            # convert default_* names
            for meta in funcmetalist:
                funcmeta.update({_from_def.get(k, k): v for k, v in meta.items()})

        reduced = {k:v for k, v in funcmeta.items() if k in _meta_keep}
        if len(reduced) < len(funcmeta):
            issue_warning("The following metadata entries were ignored: "
                          f"{sorted(set(funcmeta).difference(reduced))}.", prefix=self.msginfo)
            funcmeta = reduced

        use_jax = compmeta.get('use_jax', False)

        # first, retrieve inputs from the function signature
        sig = inspect.signature(func)
        for name, p in sig.parameters.items():
            ins[name] = meta = {'val': None, 'units': None, 'shape': None}
            # start with component wide metadata
            meta.update((k,v) for k, v in compmeta.items() if k in _meta_keep)
            meta.update(funcmeta)  # override with function wide metadata
            if p.annotation is not inspect.Parameter.empty:
                if isinstance(p.annotation, dict):
                    # Finally, override with variable specific metadata
                    meta.update(p.annotation)
                else:
                    raise TypeError(f"{self.msginfo}: Input '{name}' annotation should be a dict, "
                                    f"but is type '{type(p.annotation).__name__}'.")

            if p.default is not inspect._empty:
                if meta['val'] is not None:
                    issue_warning(f"Default value for function input '{name}' overrides 'val' set "
                                  "in the annotation.", prefix=self.msginfo)
                meta['val'] = p.default

            # assume a default value if necessary
            if meta['val'] is None and meta['shape'] is None:
                meta['val'] = default_val

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
                        raise ValueError(f"{self.msginfo}: Input '{name}' default value has shape "
                                         f"{shape}, but shape was specified as {meta['shape']}.")

        outmeta = {}
        if sig.return_annotation is not inspect.Signature.empty:
            outmeta.update(sig.return_annotation)

        # Parse the function code to possibly identify the names of the return values and input/output
        # dependencies. Return names will be non-None only if they are simple name, e.g.,
        #  return a, b, c
        outlist = []
        try:
            ret_info = get_function_deps(func)
        except RuntimeError:
            #  this could happen if function is compiled or has multiple return lines
            if not outmeta:
                raise RuntimeError(f"{self.msginfo}: Couldn't determine function return names or "
                                   "number of return values based on AST and no return value "
                                   "annotations were supplied.")
            issue_warning("Couldn't determine function return names based on AST.  Assuming number "
                          "of return values matches number of return value annotations.",
                          prefix=self.msginfo)
            outlist = list(outmeta.items())
        else:
            for o, deps in ret_info:
                if o is not None and '.' in o:  # don't allow dots in return value names
                    issue_warning(f"Ignoring return name '{o}' because names containing '.' are "
                                  "not supported.", prefix=self.msginfo)
                    o = None
                outlist.append([o, {'deps': deps}])

        notfound = []
        for i, (oname, ometa) in enumerate(outmeta.items()):
            for n, meta in outlist:
                if n == oname:
                    if meta is not ometa:
                        meta.update(ometa)
                    break
            else:  # didn't find oname
                notfound.append(oname)

        if notfound:  # try to fill in the unnamed slots with annotated output data
            inones = [i for i, (n, m) in enumerate(outlist) if n is None]  # indices with name of None
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

        return ins, outs


def _multi_callable(annotations, subname, kwgs):
    """
    Update the function annotation data in 'subname' with our named args.

    Parameters
    ----------
    annotations : dict
        Function annotation dict.
    subname : str
        Name of subdict within annotations dict.
    kwgs : dict
        Keyword args dict passed into decorator.
    """
    if ':meta' not in annotations:
        annotations[':meta'] = {}
    if subname not in annotations[':meta']:
        annotations[':meta'][subname] = []
    lst = annotations[':meta'][subname]
    # decorators are called inside out, so put new one in first entry
    annotations[':meta'][subname] = [kwgs] + lst


class func_meta(object):
    """
    Update a function's annotation data with uniform defaults.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.
    """

    def __init__(self, **kwargs):
        """
        Copy the named args passed in.
        """
        self._kwargs = kwargs

    def __call__(self, fn):
        """
        Update the function's annotation data with our named args.

        Parameters
        ----------
        fn : function
            Update the annotation data for this function.

        Returns
        -------
        function
            The original function with modified annotation data.
        """
        _multi_callable(fn.__annotations__, 'func_meta', self._kwargs)
        return fn


class declare_partials(object):
    """
    Store declare_partials info in function's annotation dict.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.
    """

    def __init__(self, **kwargs):
        """
        Copy the named args passed in.
        """
        self._kwargs = kwargs

    def __call__(self, fn):
        """
        Update the function's 'declare_partials' data with our named args.

        Parameters
        ----------
        fn : function
            Update the annotation data for this function.

        Returns
        -------
        function
            The original function with modified annotation data.
        """
        _multi_callable(fn.__annotations__, 'declare_partials', self._kwargs)
        return fn


class declare_coloring(object):
    """
    Store declare_coloring info in function's annotation dict.

    Parameters
    ----------
    **kwargs : dict
        Named args passed to the decorator.
    """

    def __init__(self, **kwargs):
        """
        Copy the named args passed in.
        """
        self._kwargs = kwargs

    def __call__(self, fn):
        """
        Update the function's 'declare_coloring' data with our named args.

        Parameters
        ----------
        fn : function
            Update the annotation data for this function.

        Returns
        -------
        function
            The original function with modified annotation data.
        """
        _multi_callable(fn.__annotations__, 'declare_coloring', self._kwargs)
        return fn
