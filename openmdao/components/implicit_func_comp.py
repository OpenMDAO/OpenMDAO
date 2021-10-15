"""Define the ImplicitFuncComp class."""


import numpy as np
from numpy import asarray, isscalar
from itertools import chain
from openmdao.core.explicitcomponent import ImplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import setup_func_comp_io, fill_vector


class ImplicitFuncComp(ImplicitComponent):
    """
    An implicit component that wraps a python function.

    Parameters
    ----------
    func : function
        The function to be wrapped by this Component.
    **kwargs : named args
        Args passed down to ImplicitComponent.

    Attributes
    ----------
    _func : callable
        The function wrapper used by this component.
    """

    def __init__(self, apply_nonlinear, solve_nonlinear=None, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._apply_nonlinear_func = omf.wrap(apply_nonlinear)

    def setup(self):
        """
        Define out inputs and outputs.
        """
        setup_func_comp_io(self)

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

    def apply_nonlinear(self, inputs, outputs, residuals):
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
        """
        fill_vector(residuals,
                    self._apply_nonlinear_func(*chain(inputs.values(), outputs.values())))

    def _solve_nonlinear(self, inputs, outputs):
        """
        Use numpy to solve Ax=b for x.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        self._solve_nonlin_func()
        # self._lup = linalg.lu_factor(inputs['A'])
        # outputs['x'] = linalg.lu_solve(self._lup, inputs['b'])
        pass

    def linearize(self, inputs, outputs, jacobian):
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
        # J['x', 'A'] = np.tile(x, size).flat
        # J['x', 'x'] = np.tile(inputs['A'].flat, vec_size)
        pass
