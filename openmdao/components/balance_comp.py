"""Define the BalanceComp class."""

from __future__ import print_function, division, absolute_import

from numbers import Number
from six import iteritems

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent


class BalanceComp(ImplicitComponent):
    """
    A simple equation balance for solving implicit equations.
    """

    def __init__(self, name=None, eq_units=None, lhs_name=None,
                 rhs_name=None, rhs_val=0.0, mult_name=None, mult_val=1.0, **kwargs):
        r"""
        Initialize a BalanceComp, optionally creating a new implicit state variable.

        The BalanceComp is a bit like IndepVarComp in that it allows for the
        creation of one or more implicit state variables, and computes the residuals
        for those variables based on the following equation.

        .. math::

          f_{mult}(x,...) \times f_{lhs}(x,...) = f_{rhs}(x,...)

        Where :math:`f_{lhs}` represents the left-hand-side of the equation,
        :math:`f_{rhs}` represents the right-hand-side, and :math:`f_{mult}`
        is an optional multiplier on the left hand side.  At least one of these
        quantities should be a function of the associated state variable.  If left
        unconnected the multiplier is simply 1.0.

        New state variables, and their associated residuals are created by
        calling `add_balance`.  As an example, solving the equation
        :math:`x**2 = 2` implicitly can be be accomplished as follows:

        ::
            prob = Problem(model=Group())

            bal = BalanceComp()

            bal.add_balance('x', val=1.0)

            tgt = IndepVarComp(name='y_tgt', val=2)

            exec_comp = ExecComp('y=x**2')

            prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

            prob.model.add_subsystem(name='exec', subsys=exec_comp)

            prob.model.add_subsystem(name='balance', subsys=bal)

            prob.model.connect('y_tgt', 'balance.rhs:x')
            prob.model.connect('balance.x', 'exec.x')
            prob.model.connect('exec.y', 'balance.lhs:x')

            prob.model.linear_solver = DirectSolver()
            prob.model.nonlinear_solver = NewtonSolver()

            prob.setup()

            prob.run_model()

        The arguments to add_balance can be provided on initialization to provide a balance
        with a one state/residual without the need to call `add_output`:

        ::
            prob = Problem(model=Group())

            bal = BalanceComp('x', val=1.0)

            tgt = IndepVarComp(name='y_tgt', val=2)

            exec_comp = ExecComp('y=x**2')

            prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

            prob.model.add_subsystem(name='exec', subsys=exec_comp)

            prob.model.add_subsystem(name='balance', subsys=bal)

            prob.model.connect('y_tgt', 'balance.rhs:x')
            prob.model.connect('balance.x', 'exec.x')
            prob.model.connect('exec.y', 'balance.lhs:x')

            prob.model.linear_solver = DirectSolver()
            prob.model.nonlinear_solver = NewtonSolver()

            prob.setup()

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
            with the shape (optionally) given by the val option in kwargs.
        mult_name : str or None
            Optional name for the LHS multiplier variable associated with the implicit state
            variable. If None, the default will be used: 'mult:{name}'.
        mult_val : int, float, or np.array
            Default value for the LHS multiplier of the given state.  Must be compatible
            with the shape (optionally) given by the val option in kwargs.
        kwargs : dict
            Additional arguments to be passed for the creation of the implicit state variable.
        """
        super(BalanceComp, self).__init__()
        self._state_vars = {}
        if name is not None:
            self.add_balance(name, eq_units, lhs_name, rhs_name, rhs_val,
                             mult_name, mult_val, **kwargs)

    def setup(self):
        """
        Define the independent variables, output variables, and partials.
        """
        self.declare_partials(of='*', wrt='*', dependent=False)

        for name, options in iteritems(self._state_vars):

            for s in ('lhs', 'rhs', 'mult'):
                if options['{0}_name'.format(s)] is None:
                    options['{0}_name'.format(s)] = '{0}:{1}'.format(s, name)

            val = options['kwargs'].get('val', np.ones(1))
            if isinstance(val, Number):
                n = 1
            else:
                n = len(val)
            self._state_vars[name]['size'] = n

            self.add_output(name, **options['kwargs'])

            self.add_input(options['lhs_name'],
                           val=np.ones(n),
                           units=options['eq_units'])

            self.add_input(options['rhs_name'],
                           val=options['rhs_val'] * np.ones(n),
                           units=options['eq_units'])

            self.add_input(options['mult_name'],
                           val=options['mult_val'] * np.ones(n),
                           units=None)

            self._scale_factor = np.ones(n)
            self._dscale_drhs = np.ones(n)

            ar = np.arange(n)
            self.declare_partials(of=name, wrt=options['lhs_name'], rows=ar, cols=ar, val=1.0)
            self.declare_partials(of=name, wrt=options['rhs_name'], rows=ar, cols=ar, val=1.0)
            self.declare_partials(of=name, wrt=options['mult_name'], rows=ar, cols=ar, val=1.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Calculate the residual for each balance.
        """
        for name, options in iteritems(self._state_vars):
            lhs_name = options['lhs_name']
            rhs_name = options['rhs_name']
            mult_name = options['mult_name']

            lhs = inputs[lhs_name]
            rhs = inputs[rhs_name]
            mult = inputs[mult_name]

            # Indices where the rhs is near zero or not near zero
            idxs_nz = np.where(np.abs(rhs) < 2)[0]
            idxs_nnz = np.where(np.abs(rhs) >= 2)[0]

            # Compute scaling factors
            # scale factor that normalizes by the rhs, except near 0
            self._scale_factor[idxs_nnz] = 1.0 / np.abs(rhs[idxs_nnz])
            self._scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

            residuals[name] = (mult * lhs - rhs) * self._scale_factor

    def linearize(self, inputs, outputs, jacobian):
        """
        Calculate the partials of the residual for each balance.
        """
        for name, options in iteritems(self._state_vars):
            lhs_name = options['lhs_name']
            rhs_name = options['rhs_name']
            mult_name = options['mult_name']

            lhs = inputs[lhs_name]
            rhs = inputs[rhs_name]
            mult = inputs[mult_name]

            # Indices where the rhs is near zero or not near zero
            idxs_nz = np.where(np.abs(rhs) < 2)[0]
            idxs_nnz = np.where(np.abs(rhs) >= 2)[0]

            # scale factor that normalizes by the rhs, except near 0
            self._scale_factor[idxs_nnz] = 1.0 / np.abs(rhs[idxs_nnz])
            self._scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

            self._dscale_drhs[idxs_nnz] = -np.sign(rhs[idxs_nnz]) / rhs[idxs_nnz]**2
            self._dscale_drhs[idxs_nz] = -.5 * rhs[idxs_nz] / (.25 * rhs[idxs_nz] ** 2 + 1) ** 2

            # Partials of residual wrt rhs
            jacobian[name, rhs_name] = (mult * lhs - rhs) * self._dscale_drhs - self._scale_factor

            # Partials of residual wrt lhs
            jacobian[name, lhs_name] = mult * self._scale_factor

            # Partials of residual wrt mult
            jacobian[name, mult_name] = lhs * self._scale_factor

    def add_balance(self, name, eq_units=None, lhs_name=None,
                    rhs_name=None, rhs_val=0.0, mult_name=None, mult_val=1.0, **kwargs):
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
            given by the val option in kwargs.
        mult_name : str or None
            Optional name for the LHS multiplier variable associated with the implicit state
            variable. If None, the default will be used: 'mult:{name}'.
        mult_val : int, float, or np.array
            Default value for the LHS multiplier.  Must be compatible with the shape (optionally)
            given by the val option in kwargs.
        kwargs : dict
            Additional arguments to be passed for the creation of the implicit state variable.
        """
        self._state_vars[name] = {'kwargs': kwargs,
                                  'eq_units': eq_units,
                                  'lhs_name': lhs_name,
                                  'rhs_name': rhs_name,
                                  'rhs_val': rhs_val,
                                  'mult_name': mult_name,
                                  'mult_val': mult_val}
