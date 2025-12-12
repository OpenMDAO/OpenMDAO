"""Define the BalanceComp class."""

from types import FunctionType

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils import cs_safe
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.general_utils import ensure_compatible


class BalanceComp(ImplicitComponent):
    """
    A simple equation balance for solving implicit equations.

    Parameters
    ----------
        name : str
            The name of the state variable to be created.
        eq_units : str or None
            Units for the left-hand-side and right-hand-side of the equation to be balanced.
        lhs_name : str or None
            Optional name for the LHS variable associated with the implicit state variable.  If
            None, the default will be used:  'lhs:{name}'.
        rhs_name : str or None
            Optional name for the RHS variable associated with the implicit state variable.  If
            None, the default will be used:  'rhs:{name}'.
        rhs_val : int, float, or np.array
            Default value for the RHS.  Must be compatible with the shape (optionally)
            given by the val or shape option in kwargs.
        use_mult : bool
            Specifies whether the LHS multiplier is to be used.  If True, then an additional
            input `mult_name` is created, with the default value given by `mult_val`, that
            multiplies lhs.  Default is False.
        mult_name : str or None
            Optional name for the LHS multiplier variable associated with the implicit state
            variable. If None, the default will be used: 'mult:{name}'.
        mult_val : int, float, or np.array
            Default value for the LHS multiplier.  Must be compatible with the shape (optionally)
            given by the val or shape option in kwargs.
        normalize : bool
            Specifies whether or not the resulting residual should be normalized by a quadratic
            function of the RHS.
        val : float, int, or np.ndarray
            Set initial value for the state.
        lhs_kwargs : dict
            Keyword arguments to be passed to the add_input call for the left-hand-side input
            of the equation (see `add_input` method).
        rhs_kwargs : dict
            Keyword arguments to be passed to the add_input call for the right-hand-side input
            of the equation (see `add_input` method).
        mult_kwargs : dict
            Keyword arguments to be passed to the add_input call for the multiplier input
            of the equation, if used (see `add_input` method).
        **kwargs : dict
            Additional arguments to be passed for the creation of the implicit state variable.
            (see `add_output` method).

    Attributes
    ----------
    _state_vars : dict
        Cache the data provided during `add_balance`
        so everything can be saved until setup is called.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('guess_func', types=FunctionType, allow_none=True, default=None,
                             recordable=False, desc='A callable function in the form '
                             'f(inputs, outputs, residuals) that can provide an initial "guess" '
                             'value of the state variable(s) based on the inputs, outputs and '
                             'residuals.')

    def __init__(self, name=None, eq_units=None, lhs_name=None, rhs_name=None, rhs_val=0.0,
                 use_mult=False, mult_name=None, mult_val=1.0, normalize=True, val=None,
                 lhs_kwargs=None, rhs_kwargs=None, mult_kwargs=None, **kwargs):
        r"""
        Initialize a BalanceComp, optionally creating a new implicit state variable.

        The BalanceComp allows for the creation of one or more implicit state variables,
        and computes the residuals for those variables based on the following equation.

        .. math::

            \mathcal{R}_{name} =
            \frac{f_{mult}(x,...) \times f_{lhs}(x,...) - f_{rhs}(x,...)}{f_{norm}(f_{rhs}(x,...))}


        Where :math:`f_{lhs}` represents the left-hand-side of the equation,
        :math:`f_{rhs}` represents the right-hand-side, and :math:`f_{mult}`
        is an optional multiplier on the left hand side.  At least one of these
        quantities should be a function of the associated state variable.  If
        use_mult is True the default value of the multiplier is 1. The optional normalization
        function :math:`f_{norm}(f_{rhs}(x,...))` is computed as:

        .. math::

          f_{norm}(f_{rhs}(x,...)) =
          \begin{cases}
           \left| f_{rhs} \right|, & \text{if normalize and } \left| f_{rhs} \right| \geq 2 \\
           0.25 f_{rhs}^2 + 1,     & \text{if normalize and } \left| f_{rhs} \right| < 2 \\
           1,                      & \text{if not normalize}
          \end{cases}

        New state variables, and their associated residuals are created by
        calling `add_balance`.  As an example, solving the equation
        :math:`x**2 = 2` implicitly can be be accomplished as follows:

        .. code-block:: python

            prob = Problem()
            bal = BalanceComp()
            bal.add_balance('x', val=1.0)
            exec_comp = ExecComp('y=x**2')
            prob.model.add_subsystem(name='exec', subsys=exec_comp)
            prob.model.add_subsystem(name='balance', subsys=bal)
            prob.model.connect('balance.x', 'exec.x')
            prob.model.connect('exec.y', 'balance.lhs:x')
            prob.model.linear_solver = DirectSolver()
            prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)
            prob.setup()
            prob.set_val('exec.x', 2)
            prob.run_model()

        The arguments to add_balance can be provided on initialization to provide a balance
        with a one state/residual without the need to call `add_balance`:

        .. code-block:: python

            prob = Problem()
            bal = BalanceComp('x', val=1.0)
            exec_comp = ExecComp('y=x**2')
            prob.model.add_subsystem(name='exec', subsys=exec_comp)
            prob.model.add_subsystem(name='balance', subsys=bal)
            prob.model.connect('balance.x', 'exec.x')
            prob.model.connect('exec.y', 'balance.lhs:x')
            prob.model.linear_solver = DirectSolver()
            prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)
            prob.setup()
            prob.set_val('exec.x', 2)
            prob.run_model()
        """
        # Pre-declare options so we can separate component kwargs from output kwargs.
        self.options = OptionsDictionary()
        self._declare_options()
        comp_kwargs = set(self.options._dict.keys())
        super().__init__(**{k: v for k, v in kwargs.items() if k in comp_kwargs})

        self._state_vars = {}

        if name is not None:
            _kwargs = {k: v for k, v in kwargs.items() if k not in comp_kwargs}
            self.add_balance(name, eq_units=eq_units, lhs_name=lhs_name,
                             rhs_name=rhs_name, rhs_val=rhs_val, use_mult=use_mult,
                             mult_name=mult_name, mult_val=mult_val, normalize=normalize,
                             val=val, lhs_kwargs=lhs_kwargs, rhs_kwargs=lhs_kwargs,
                             mult_kwargs=mult_kwargs, **_kwargs)

        self._no_check_partials = True

    def _declare_options(self):
        super()._declare_options()
        self.options.declare('guess_func', types=FunctionType, allow_none=True, default=None,
                             recordable=False, desc='A callable function in the form '
                             'f(inputs, outputs, residuals) that can provide an initial "guess" '
                             'value of the state variable(s) based on the inputs, outputs and '
                             'residuals.')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Calculate the residual for each balance.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        """
        for name, options in self._state_vars.items():
            lhs = inputs[options['lhs_name']]
            rhs = inputs[options['rhs_name']]

            # set dtype to rhs.dtype to prevent
            # "Casting complex values to real discards the imaginary part" warning.
            _scale_factor = np.ones((rhs.shape), dtype=rhs.dtype)

            if options['normalize']:

                # Indices where the rhs is near zero or not near zero
                absrhs = cs_safe.abs(rhs)
                if rhs.shape == ():
                    if absrhs < 2:
                        _scale_factor = 1.0 / (.25 * rhs**2 + 1)
                    else:
                        _scale_factor = 1.0 / absrhs
                else:
                    idxs_nz = np.where(absrhs < 2)
                    idxs_nnz = np.where(absrhs >= 2)

                    # Compute scaling factors
                    # scale factor that normalizes by the rhs, except near 0
                    _scale_factor[idxs_nnz] = 1.0 / absrhs[idxs_nnz]
                    _scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

            if options['use_mult']:
                residuals[name] = (inputs[options['mult_name']] * lhs - rhs) * _scale_factor
            else:
                residuals[name] = (lhs - rhs) * _scale_factor

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
        for name, options in self._state_vars.items():
            lhs_name = options['lhs_name']
            rhs_name = options['rhs_name']

            lhs = inputs[lhs_name]
            rhs = inputs[rhs_name]

            _scale_factor = np.ones((rhs.shape), dtype=rhs.dtype)
            _dscale_drhs = np.zeros((rhs.shape), dtype=rhs.dtype)

            if options['normalize']:
                absrhs = cs_safe.abs(rhs)
                if rhs.shape == ():
                    if absrhs < 2:
                        _scale_factor = 1.0 / (.25 * rhs**2 + 1)
                        _dscale_drhs = -.5 * rhs / (.25 * rhs**2 + 1) ** 2
                    else:
                        _scale_factor = 1.0 / absrhs
                        _dscale_drhs = -np.sign(rhs) / rhs**2
                else:
                    # Indices where the rhs is near zero or not near zero
                    idxs_nz = np.where(absrhs < 2)[0]
                    idxs_nnz = np.where(absrhs >= 2)[0]

                    # scale factor that normalizes by the rhs, except near 0
                    _scale_factor[idxs_nnz] = 1.0 / absrhs[idxs_nnz]
                    _scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

                    _dscale_drhs[idxs_nnz] = -np.sign(rhs[idxs_nnz]) / rhs[idxs_nnz] ** 2
                    _dscale_drhs[idxs_nz] = -.5 * rhs[idxs_nz] / (.25 * rhs[idxs_nz] ** 2 + 1) ** 2

            if options['use_mult']:
                mult_name = options['mult_name']
                mult = inputs[mult_name]

                # Partials of residual wrt mult
                deriv = lhs * _scale_factor
                jacobian[name, mult_name] = deriv.flatten()
            else:
                mult = 1.0

            # Partials of residual wrt rhs
            deriv = (mult * lhs - rhs) * _dscale_drhs - _scale_factor
            jacobian[name, rhs_name] = deriv.flatten()

            # Partials of residual wrt lhs
            deriv = mult * _scale_factor
            jacobian[name, lhs_name] = deriv.flatten()

    def guess_nonlinear(self, inputs, outputs, residuals):
        """
        Provide initial guess for states.

        Override this method to set the initial guess for states.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        """
        if self.options['guess_func'] is not None:
            self.options['guess_func'](inputs, outputs, residuals)

    def add_balance(self, name, eq_units=None, lhs_name=None, rhs_name=None, rhs_val=0.0,
                    use_mult=False, mult_name=None, mult_val=1.0, normalize=True, val=None,
                    lhs_kwargs=None, rhs_kwargs=None, mult_kwargs=None, **kwargs):
        """
        Add a new state variable and associated equation to be balanced.

        This will create new inputs `lhs:name`, `rhs:name`, and `mult:name` that will
        define the left and right sides of the equation to be balanced, and a
        multiplier for the left-hand-side.

        Parameters
        ----------
        name : str
            The name of the state variable to be created.
        eq_units : str or None
            Units for the left-hand-side and right-hand-side of the equation to be balanced.
        lhs_name : str or None
            Optional name for the LHS variable associated with the implicit state variable.  If
            None, the default will be used:  'lhs:{name}'.
        rhs_name : str or None
            Optional name for the RHS variable associated with the implicit state variable.  If
            None, the default will be used:  'rhs:{name}'.
        rhs_val : int, float, or np.array
            Default value for the RHS.  Must be compatible with the shape (optionally)
            given by the val or shape option in kwargs.
        use_mult : bool
            Specifies whether the LHS multiplier is to be used.  If True, then an additional
            input `mult_name` is created, with the default value given by `mult_val`, that
            multiplies lhs.  Default is False.
        mult_name : str or None
            Optional name for the LHS multiplier variable associated with the implicit state
            variable. If None, the default will be used: 'mult:{name}'.
        mult_val : int, float, or np.array
            Default value for the LHS multiplier.  Must be compatible with the shape (optionally)
            given by the val or shape option in kwargs.
        normalize : bool
            Specifies whether or not the resulting residual should be normalized by a quadratic
            function of the RHS.
        val : float, int, or np.ndarray
            Set initial value for the state.
        lhs_kwargs : dict
            Keyword arguments to be passed to the add_input call for the left-hand-side input
            of the equation (see `add_input` method).
        rhs_kwargs : dict
            Keyword arguments to be passed to the add_input call for the right-hand-side input
            of the equation (see `add_input` method).
        mult_kwargs : dict
            Keyword arguments to be passed to the add_input call for the multiplier input
            of the equation, if used (see `add_input` method).
        **kwargs : dict
            Additional arguments to be passed for the creation of the implicit state variable.
            (see `add_output` method).
        """
        options = {'name': name,
                   'eq_units': eq_units,
                   'use_mult': use_mult,
                   'normalize': normalize}

        lhs_kwargs = lhs_kwargs or {}
        rhs_kwargs = rhs_kwargs or {}
        mult_kwargs = mult_kwargs or {}
        output_kwargs = kwargs

        # Put the legacy arguments in the kwarg dictionaries
        lhs_kwargs['name'] = options['lhs_name'] = lhs_kwargs.get('name',
                                                                  lhs_name or f'lhs:{name}')
        rhs_kwargs['name'] = options['rhs_name'] = rhs_kwargs.get('name',
                                                                  rhs_name or f'rhs:{name}')
        mult_kwargs['name'] = options['mult_name'] = mult_kwargs.get('name',
                                                                     mult_name or f'mult:{name}')

        lhs_kwargs['units'] = lhs_kwargs.get('units', eq_units)
        rhs_kwargs['units'] = rhs_kwargs.get('units', eq_units)

        rhs_kwargs['val'] = rhs_val = rhs_kwargs.get('val', rhs_val)
        mult_kwargs['val'] = mult_kwargs.get('val', mult_val)

        # Store options
        self._state_vars[name] = options

        if val is None:
            # If user doesn't specify initial guess for val, we can size problem from initial
            # rhs_val.
            if 'shape' not in output_kwargs and np.ndim(rhs_val) > 0:
                output_kwargs['shape'] = rhs_val.shape
            else:
                output_kwargs['val'] = 1.0
        else:
            output_kwargs['val'] = val

        output_has_explicit_shape = 'shape' in output_kwargs or \
            np.ndim(output_kwargs.get('val', 0)) > 0
        rhs_has_explicit_shape = 'shape' in rhs_kwargs or \
            np.ndim(rhs_kwargs.get('val', 0)) > 0

        shape = None
        if output_has_explicit_shape:
            _, shape = ensure_compatible(name, output_kwargs.get('val', 1.),
                                         shape=output_kwargs.get('shape', None),
                                         default_shape=self.options['default_shape'])
        elif rhs_has_explicit_shape:
            _, shape = ensure_compatible(name, rhs_kwargs.get('val', 1.),
                                         shape=rhs_kwargs.get('shape', None),
                                         default_shape=self.options['default_shape'])

        if shape is not None:
            for _kwargs in (output_kwargs, lhs_kwargs, rhs_kwargs, mult_kwargs):
                _kwargs['shape'] = shape

        self.add_output(name, **output_kwargs)
        self.add_input(**lhs_kwargs)
        self.add_input(**rhs_kwargs)

        if use_mult:
            self.add_input(**mult_kwargs)

    def setup_partials(self):
        """
        Declare the partials for outputs once all variable shapes are known.
        """
        for name, options in self._state_vars.items():
            self.declare_partials(of=name, wrt=options['lhs_name'], diagonal=True, val=1.0)
            self.declare_partials(of=name, wrt=options['rhs_name'], diagonal=True, val=1.0)

            if options['use_mult']:
                self.declare_partials(of=name, wrt=options['mult_name'], diagonal=True, val=1.0)
