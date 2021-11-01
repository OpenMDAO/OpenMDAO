"""Define the FuncComponent class."""

try:
    import jax
    from jax import jvp, vjp, vmap, random, jit
    import jax.numpy as jnp
except ImportError:
    jax = None

import re
import numpy as np
from numpy import asarray, isscalar, imag, complex as npcomplex
from itertools import product
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
from openmdao.utils.units import valid_units
from openmdao.utils.om_warnings import issue_warning
import openmdao.func_api as omf


# regex to check for variable names.
namecheck_rgx = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

# Names that are not allowed for input or output variables (keywords for options)
_disallowed_varnames = {'units', 'shape', 'shape_by_conn', 'run_root_only', 'distributed',
                        'assembled_jac_type'}

_meta_keep = {'units', 'shape', 'val'}
_from_def = {'default_units': 'units', 'default_shape': 'shape'}


def _check_units_option(option, value):
    if value is not None and not valid_units(value):
        raise ValueError(f"The units '{value}' are invalid.")


def _copy_with_ignore(dct, keepers, ignore=()):
    """
    Copy the entries in the given dict whose keys are in keepers.

    Parameters
    ----------
    dct : dict
        The dictionary to be copied.
    keepers : set-like
        Set of keys for entries we want to keep.
    ignore : set or tuple
        Ignore these keys.

    Returns
    -------
    dict
        A new dict containing 'keepers' entries.
    """
    kept = {}
    warn = set()
    for k, v in dct.items():
        if k in keepers:
            kept[k] = v
        elif k not in ignore:
            warn.add(k)

    if warn:
        issue_warning(f"The following metadata entries were ignored: {sorted(warn)}.")

    return kept


class ExplicitFuncComp(ExplicitComponent):
    """
    A component that wraps a python function.

    Parameters
    ----------
    func : function
        The function to be wrapped by this Component.
    compute_partials : function or None
        If not None, call this function when computing partials.
    **kwargs : named args
        Args passed down to ExplicitComponent.

    Attributes
    ----------
    _func : callable
        The function wrapper used by this component.
    _compute_partials : function or None
        If not None, call this function when computing partials.
    """

    def __init__(self, func, compute_partials=None, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._func = omf.wrap(func)
        self._compute_partials = compute_partials

    def setup(self):
        """
        Define out inputs and outputs.
        """
        optignore = {'is_option'}

        for name, meta in self._func.get_input_meta():
            self._check_var_name(name)
            if 'is_option' in meta and meta['is_option']:
                kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                           ignore=optignore)
                self.options.declare(name, **kwargs)
            else:
                kwargs = _copy_with_ignore(meta, omf._allowed_add_input_args)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._func.get_output_meta()):
            if name is None:
                raise RuntimeError(f"{self.msginfo}: Can't add output corresponding to return "
                                   f"value in position {i} because it has no name.  Specify the "
                                   "name by returning a variable, for example 'return myvar', or "
                                   "include the name in the function's metadata.")
            self._check_var_name(name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args)
            self.add_output(name, **kwargs)

    def declare_partials(self, *args, **kwargs):
        """
        Declare information about this component's subjacobians.

        Parameters
        ----------
        *args : list
            Positional args to be passed to base class version of declare_partials.
        **kwargs : dict
            Keyword args  to be passed to base class version of declare_partials.

        Returns
        -------
        dict
            Metadata dict for the specified partial(s).
        """
        if self._compute_partials is None and ('method' not in kwargs or
                                               kwargs['method'] == 'exact'):
            raise RuntimeError(f"{self.msginfo}: declare_partials must be called with method equal "
                               "to 'cs', 'fd', or 'jax'.")

        return super().declare_partials(*args, **kwargs)

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        kwargs = self._func.get_declare_coloring()
        if kwargs is not None:
            self.declare_coloring(**kwargs)

        for kwargs in self._func.get_declare_partials():
            self.declare_partials(**kwargs)

        super()._setup_partials()

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name].
        """
        if self._compute_partials is None:
            return

        args = list(inputs.values())
        args.append(partials)
        self._compute_partials(*args)

    def _check_var_name(self, name):
        match = namecheck_rgx.match(name)
        if match is None or match.group() != name:
            raise NameError(f"{self.msginfo}: '{name}' is not a valid variable name.")

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
