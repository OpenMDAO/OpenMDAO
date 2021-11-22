"""Define the FuncComponent class."""

import functools
import numpy as np
from numpy import asarray, isscalar
from itertools import chain
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options


class ExplicitFuncComp(ExplicitComponent):
    """
    A component that wraps a python function.

    Parameters
    ----------
    compute : function
        The function to be wrapped by this Component.
    compute_partials : function or None
        If not None, call this function when computing partials.
    **kwargs : named args
        Args passed down to ExplicitComponent.

    Attributes
    ----------
    _compute : callable
        The function wrapper used by this component.
    _compute_partials : function or None
        If not None, call this function when computing partials.
    """

    def __init__(self, compute, compute_partials=None, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._compute = omf.wrap(compute)
        # in case we're doing jit, force setup of wrapped func because we compute output shapes
        # during setup and that won't work on a jit compiled function
        if self._compute._call_setup:
            self._compute._setup()

        self._compute_partials = compute_partials

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        _add_options(self)

    def setup(self):
        """
        Define out inputs and outputs.
        """
        optignore = {'is_option'}

        for name, meta in self._compute.get_input_meta():
            _check_var_name(self, name)
            if 'is_option' in meta and meta['is_option']:
                kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                           ignore=optignore)
                self.options.declare(name, **kwargs)
            else:
                kwargs = omf._filter_dict(meta, omf._allowed_add_input_args)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._compute.get_output_meta()):
            _check_var_name(self, name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args, ignore=('resid',))
            self.add_output(name, **kwargs)

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
        outputs.set_vals(self._compute(*self._func_values(inputs)))

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
        return super().declare_partials(*args, **kwargs)

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        for kwargs in self._compute.get_declare_partials():
            self.declare_partials(**kwargs)

        kwargs = self._compute.get_declare_coloring()
        if kwargs is not None:
            self.declare_coloring(**kwargs)

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

        self._compute_partials(*self._func_values(inputs), partials)

    def _func_values(self, inputs):
        """
        Yield current function input args.

        Parameters
        ----------
        inputs : Vector
            The input vector.

        Yields
        ------
        object
            Value of current function input variable.
        """
        inps = inputs.values()

        for name, meta in self._compute._inputs.items():
            if 'is_option' in meta:
                yield self.options[name]
            else:
                yield next(inps)
