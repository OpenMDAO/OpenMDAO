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

    def add_input(self, name, val=1.0, shape=None, units=None, desc='', tags=None,
                  shape_by_conn=False, copy_shape=None, compute_shape=None, distributed=None,
                  ref=None):
        """
        Add an input to be used as a residual.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or list or tuple or ndarray or Iterable
            The initial value of the variable being added in user-defined units.
            Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array. Default is None.
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
        compute_shape : function
            A function taking a dict arg containing names and shapes of this component's outputs
            and returning the shape of this input.
        distributed : bool
            If True, this variable is a distributed variable, so it can have different sizes/values
            across MPI processes.
        ref : float or ndarray or None
            Scaling parameter. The value in the user-defined units of this residual
            when the scaled value is 1. Default is 1.
        """
        self._refs[name] = ref
        super().add_input(name, val=val, shape=shape, units=units, desc=desc, tags=tags,
                          shape_by_conn=shape_by_conn, copy_shape=copy_shape,
                          compute_shape=compute_shape, distributed=distributed)

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
        size = self._get_var_meta('x', 'size')
        rng = np.arange(size)
        self.declare_partials('y', 'x', rows=rng, cols=rng, val=3.0)
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
