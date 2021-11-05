"""Define the ImplicitFuncComp class."""

from itertools import chain
import numpy as np
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options


class ImplicitFuncComp(ImplicitComponent):
    """
    An implicit component that wraps a python function.

    Parameters
    ----------
    apply_nonlinear : function
        The function to be wrapped by this Component.
    solve_nonlinear : function or None
        Optional function to perform a nonlinear solve.
    linearize : function or None
        Optional function to compute partial derivatives.
    solve_linear : function or None
        Optional function to perform a linear solve.
    **kwargs : named args
        Args passed down to ImplicitComponent.

    Attributes
    ----------
    _apply_nonlinear_func : callable
        The function wrapper used by this component.
    _solve_nonlinear_func : function or None
        Optional function to do a nonlinear solve.
    solve_nonlinear : method
        Local override of _solve_nonlinear method.
    _solve_linear_func : function or None
        Optional function to do a linear solve.
    solve_linear : method
        Local override of solve_linear method.
    _linearize_func : function or None
        Optional function to compute partial derivatives.
    linearize : method
        Local override of linearize method.
    _linearize_info : object
        Some state information to compute in _linearize_func and pass to _solve_linear_func
    """

    def __init__(self, apply_nonlinear, solve_nonlinear=None, linearize=None, solve_linear=None,
                 **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._apply_nonlinear_func = omf.wrap(apply_nonlinear)
        self._solve_nonlinear_func = solve_nonlinear
        self._solve_linear_func = solve_linear
        self._linearize_func = linearize
        self._linearize_info = None
        if solve_nonlinear:
            self.solve_nonlinear = self._solve_nonlinear_
        if linearize:
            self.linearize = self._linearize_
        if solve_linear:
            self.solve_linear = self._solve_linear_

        if self._apply_nonlinear_func._use_jax:
            self.options['use_jax'] = True

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        _add_options(self)

    def setup(self):
        """
        Define our inputs and outputs.
        """
        optignore = {'is_option'}

        for name, meta in self._apply_nonlinear_func.get_input_meta():
            _check_var_name(self, name)
            if 'is_option' in meta and meta['is_option']:
                kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                           ignore=optignore)
                self.options.declare(name, **kwargs)
            else:
                kwargs = omf._filter_dict(meta, omf._allowed_add_input_args)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._apply_nonlinear_func.get_output_meta()):
            _check_var_name(self, name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args, ignore=('resid',))
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
        if self._linearize_func is None and ('method' not in kwargs or
                                             kwargs['method'] == 'exact'):
            raise RuntimeError(f"{self.msginfo}: declare_partials must be called with method equal "
                               "to 'cs', 'fd', or 'jax'.")

        return super().declare_partials(*args, **kwargs)

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        kwargs = self._apply_nonlinear_func.get_declare_coloring()
        if kwargs is not None:
            self.declare_coloring(**kwargs)

        for kwargs in self._apply_nonlinear_func.get_declare_partials():
            self.declare_partials(**kwargs)

        super()._setup_partials()

    def apply_nonlinear(self, inputs, outputs, residuals,
                        discrete_inputs=None, discrete_outputs=None):
        """
        R = Ax - b.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        discrete_inputs : _DictValues or None
            Dict-like object containing discrete inputs.
        discrete_outputs : _DictValues or None
            Dict-like object containing discrete outputs.
        """
        residuals.set_vals(self._apply_nonlinear_func(*self._ordered_values(inputs, outputs)))

    def _solve_nonlinear_(self, inputs, outputs):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        self._outputs.set_vals(self._solve_nonlinear_func(*self._ordered_values(inputs, outputs)))

    def _linearize_(self, inputs, outputs, jacobian):
        """
        Calculate the partials of the residual for each balance.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        jacobian : Jacobian
            Sub-jac components written to jacobian[output_name, input_name].
        """
        self._linearize_info = self._linearize_func(*chain(self._ordered_values(inputs, outputs),
                                                           (jacobian,)))

    def _solve_linear_(self, d_outputs, d_residuals, mode):
        r"""
        Run solve_linear function if there is one.

        Parameters
        ----------
        d_outputs : Vector
            Unscaled, dimensional quantities read via d_outputs[key].
        d_residuals : Vector
            Unscaled, dimensional quantities read via d_residuals[key].
        mode : str
            Derivative solutiion direction, either 'fwd' or 'rev'.
        """
        if mode == 'fwd':
            d_outputs.set_vals(self._solve_linear_func(*chain(d_residuals.values(),
                                                              (mode, self._linearize_info))))
        else:  # rev
            d_residuals.set_vals(self._solve_linear_func(*chain(d_outputs.values(),
                                                                (mode, self._linearize_info))))

    def _ordered_values(self, inputs, outputs):
        """
        Yield function input args in their proper order.

        In OpenMDAO, states are outputs, but for our some of our functions they are inputs, so
        this function yields the values of the inputs and states in the same order that they
        were originally given for the _apply_nonlinear_func.

        Parameters
        ----------
        inputs : Vector
            The input vector.
        outputs : Vector
            The output vector (contains the states).

        Yields
        ------
        float or ndarray
            Value of input or state variable.
        """
        inps = inputs.values()
        outs = outputs.values()

        for name, meta in self._apply_nonlinear_func._inputs.items():
            if 'is_option' in meta:  # it's an option
                yield self.options[name]
            elif 'resid' in meta:  # it's a state
                yield next(outs)
            else:
                yield next(inps)
