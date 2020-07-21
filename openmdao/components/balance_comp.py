"""Define the BalanceComp class."""

from types import FunctionType
from numbers import Number

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent


class BalanceComp(ImplicitComponent):
    """
    A simple equation balance for solving implicit equations.

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
                 use_mult=False, mult_name=None, mult_val=1.0, normalize=True, val=None, **kwargs):
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
            Default value for the RHS of the given state.  Must be compatible
            with the shape (optionally) given by the val or shape option in kwargs.
        use_mult : bool
            Specifies whether the LHS multiplier is to be used.  If True, then an additional
            input `mult_name` is created, with the default value given by `mult_val`, that
            multiplies lhs.  Default is False.
        mult_name : str or None
            Optional name for the LHS multiplier variable associated with the implicit state
            variable. If None, the default will be used: 'mult:{name}'.
        mult_val : int, float, or np.array
            Default value for the LHS multiplier of the given state.  Must be compatible
            with the shape (optionally) given by the val or shape option in kwargs.
        normalize : bool
            Specifies whether or not the resulting residual should be normalized by a quadratic
            function of the RHS.
        val : float, int, or np.ndarray
            Set initial value for the state.
        **kwargs : dict
            Additional arguments to be passed for the creation of the implicit state variable.
            (see `add_output` method).
        """
        if 'guess_func' in kwargs:
            super(BalanceComp, self).__init__(guess_func=kwargs['guess_func'])
            kwargs.pop('guess_func')
        else:
            super(BalanceComp, self).__init__()

        self._state_vars = {}

        if name is not None:
            self.add_balance(name, eq_units, lhs_name, rhs_name, rhs_val,
                             use_mult, mult_name, mult_val, normalize, val, **kwargs)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Calculate the residual for each balance.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        if inputs._under_complex_step:
            self._scale_factor = self._scale_factor.astype(np.complex)
        else:
            self._scale_factor = self._scale_factor.real

        for name, options in self._state_vars.items():
            lhs = inputs[options['lhs_name']]
            rhs = inputs[options['rhs_name']]

            if options['normalize']:
                # Indices where the rhs is near zero or not near zero
                idxs_nz = np.where(np.abs(rhs) < 2)[0]
                idxs_nnz = np.where(np.abs(rhs) >= 2)[0]

                # Compute scaling factors
                # scale factor that normalizes by the rhs, except near 0
                self._scale_factor[idxs_nnz] = 1.0 / np.abs(rhs[idxs_nnz])
                self._scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)
            else:
                self._scale_factor[:] = 1.0

            if options['use_mult']:
                residuals[name] = (inputs[options['mult_name']] * lhs - rhs) * self._scale_factor
            else:
                residuals[name] = (lhs - rhs) * self._scale_factor

    def linearize(self, inputs, outputs, jacobian):
        """
        Calculate the partials of the residual for each balance.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        jacobian : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """
        if inputs._under_complex_step:
            self._dscale_drhs = self._dscale_drhs.astype(np.complex)
        else:
            self._dscale_drhs = self._dscale_drhs.real

        for name, options in self._state_vars.items():
            lhs_name = options['lhs_name']
            rhs_name = options['rhs_name']

            lhs = inputs[lhs_name]
            rhs = inputs[rhs_name]

            if options['normalize']:
                # Indices where the rhs is near zero or not near zero
                idxs_nz = np.where(np.abs(rhs) < 2)[0]
                idxs_nnz = np.where(np.abs(rhs) >= 2)[0]

                # scale factor that normalizes by the rhs, except near 0
                self._scale_factor[idxs_nnz] = 1.0 / np.abs(rhs[idxs_nnz])
                self._scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

                self._dscale_drhs[idxs_nnz] = -np.sign(rhs[idxs_nnz]) / rhs[idxs_nnz]**2
                self._dscale_drhs[idxs_nz] = -.5 * rhs[idxs_nz] / (.25 * rhs[idxs_nz] ** 2 + 1) ** 2
            else:
                self._scale_factor[:] = 1.0
                self._dscale_drhs[:] = 0.0

            if options['use_mult']:
                mult_name = options['mult_name']
                mult = inputs[mult_name]

                # Partials of residual wrt mult
                deriv = lhs * self._scale_factor
                jacobian[name, mult_name] = deriv.flatten()
            else:
                mult = 1.0

            # Partials of residual wrt rhs
            deriv = (mult * lhs - rhs) * self._dscale_drhs - self._scale_factor
            jacobian[name, rhs_name] = deriv.flatten()

            # Partials of residual wrt lhs
            deriv = mult * self._scale_factor
            jacobian[name, lhs_name] = deriv.flatten()

    def guess_nonlinear(self, inputs, outputs, residuals):
        """
        Provide initial guess for states.

        Override this method to set the initial guess for states.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        if self.options['guess_func'] is not None:
            self.options['guess_func'](inputs, outputs, residuals)

    def add_balance(self, name, eq_units=None, lhs_name=None, rhs_name=None, rhs_val=0.0,
                    use_mult=False, mult_name=None, mult_val=1.0, normalize=True, val=None,
                    **kwargs):
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
        **kwargs : dict
            Additional arguments to be passed for the creation of the implicit state variable.
            (see `add_output` method).
        """
        options = {'kwargs': kwargs,
                   'eq_units': eq_units,
                   'lhs_name': lhs_name,
                   'rhs_name': rhs_name,
                   'rhs_val': rhs_val,
                   'use_mult': use_mult,
                   'mult_name': mult_name,
                   'mult_val': mult_val,
                   'normalize': normalize}

        self._state_vars[name] = options

        if val is not None:
            options['kwargs']['val'] = val

        meta = self.add_output(name, **options['kwargs'])

        shape = meta['shape']

        for s in ('lhs', 'rhs', 'mult'):
            if options['{0}_name'.format(s)] is None:
                options['{0}_name'.format(s)] = '{0}:{1}'.format(s, name)

        self.add_input(options['lhs_name'],
                       val=np.ones(shape),
                       units=options['eq_units'])

        self.add_input(options['rhs_name'],
                       val=options['rhs_val'] * np.ones(shape),
                       units=options['eq_units'])

        if options['use_mult']:
            self.add_input(options['mult_name'],
                           val=options['mult_val'] * np.ones(shape),
                           units=None)

        self._scale_factor = np.ones(shape)
        self._dscale_drhs = np.ones(shape)

        ar = np.arange(np.prod(shape))
        self.declare_partials(of=name, wrt=options['lhs_name'], rows=ar, cols=ar, val=1.0)
        self.declare_partials(of=name, wrt=options['rhs_name'], rows=ar, cols=ar, val=1.0)

        if options['use_mult']:
            self.declare_partials(of=name, wrt=options['mult_name'], rows=ar, cols=ar, val=1.0)
