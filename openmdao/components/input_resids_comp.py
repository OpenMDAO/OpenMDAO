"""InputResidsComp provides a simple implicit component with minimal boilerplate."""

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent


class InputResidsComp(ImplicitComponent):
    """
    Class definition for the InputResidsComp.

    Uses all inputs as residuals while allowing individual outputs that are not necessarily
    associated with a specific residual.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    Attributes
    ----------
    _refs : dict
        Residual ref values that are cached during calls to the overloaded add_input method.
    """

    def __init__(self, **kwargs):
        """
        Initialize the InputResidsComp.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to the __init__ method of ImplicitComponent
        """
        self._refs = {}
        super().__init__(**kwargs)

    def add_input(self, name, ref=None, **kwargs):
        """
        Add an input to be used as a residual.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        ref : float or ndarray or None
            Scaling parameter. The value in the user-defined units of this residual
            when the scaled value is 1. Default is 1.
        **kwargs : dict
            Additional arguments passed to the add_input method of ImplicitComponent.
        """
        self._refs[name] = ref
        super().add_input(name, **kwargs)

    def setup_residuals(self):
        """
        Delay calls for add_residual for this component.

        This method is used since input/residual sizes may not
        be known until final setup.
        """
        for name in self._var_rel_names['input']:
            meta = self._var_rel2meta[name]
            resid_name = f'resid_{name}'
            self.add_residual(resid_name, shape=meta['shape'], units=meta['units'],
                              desc=meta['desc'], ref=self._refs[name])

    def setup_partials(self):
        """
        Delay calls to declare_partials for the component.

        This method is used because input/residual sizes
        may not be known until final setup.
        """
        for name in self._var_rel_names['input']:
            resid_name = 'resid_' + name
            size = self._var_rel2meta[name]['size']
            ar = np.arange(size, dtype=int)
            self.declare_partials(of=resid_name, wrt=name, rows=ar, cols=ar, val=1.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        """
        residuals.set_val(inputs.asarray())
