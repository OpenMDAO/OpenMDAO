"""Define the EQConstraintComp class."""

from __future__ import print_function, division, absolute_import

from numbers import Number
from six import iteritems

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class EQConstraintComp(ExplicitComponent):
    """
    A component that computes the difference between two inputs to test for equality.

    Attributes
    ----------
    _output_vars : dict
        Cache the data provided during `add_eq_output`
        so everything can be saved until setup is called.
    """

    def __init__(self, name=None, eq_units=None, lhs_name=None, rhs_name=None, rhs_val=0.0,
                 use_mult=False, mult_name=None, mult_val=1.0, add_constraint=False,
                 **kwargs):
        r"""
        Initialize an EQConstraintComp, optionally add an output constraint to the model.

        The EQConstraintComp allows for the creation of one or more output variables and
        computes the values for those variables based on the following equation:

        .. math::

          name_{output} = name_{mult} \times name_{lhs} - name_{rhs}

        Where :math:`name_{lhs}` represents the left-hand-side of the equality,
        :math:`name_{rhs}` represents the right-hand-side, and :math:`name_{mult}`
        is an optional multiplier on the left hand side. If use_mult is True then
        the default value of the multiplier is 1.

        New output variables are created by calling `add_eq_output`.

        Parameters
        ----------
        name : str
            The name of the output variable to be created.
        eq_units : str or None
            Units for the left-hand-side and right-hand-side of the difference equation.
        lhs_name : str or None
            Optional name for the LHS variable associated with the difference equation.
            If None, the default will be used:  'lhs:{name}'.
        rhs_name : str or None
            Optional name for the RHS variable associated with the difference equation.
            If None, the default will be used:  'rhs:{name}'.
        rhs_val : int, float, or np.array
            Default value for the RHS of the given output.  Must be compatible
            with the shape (optionally) given by the val or shape option in kwargs.
        use_mult : bool
            Specifies whether the LHS multiplier is to be used.  If True, then an additional
            input `mult_name` is created, with the default value given by `mult_val`, that
            multiplies lhs.  Default is False.
        mult_name : str or None
            Optional name for the LHS multiplier variable associated with the output
            variable. If None, the default will be used: 'mult:{name}'.
        mult_val : int, float, or np.array
            Default value for the LHS multiplier of the given output.  Must be compatible
            with the shape (optionally) given by the val or shape option in kwargs.
        add_constraint : bool
            Specifies whether to add an equality constraint.
        **kwargs : dict
            Additional arguments to be passed for the creation of the output variable.
            (see `add_output` method).
        """
        super(EQConstraintComp, self).__init__()
        self._output_vars = {}
        if name is not None:
            self.add_eq_output(name, eq_units, lhs_name, rhs_name, rhs_val,
                               use_mult, mult_name, mult_val, add_constraint, **kwargs)

    def setup(self):
        """
        Define the independent variables, output variables, and partials.
        """
        for name, options in iteritems(self._output_vars):

            for s in ('lhs', 'rhs', 'mult'):
                if options['{0}_name'.format(s)] is None:
                    options['{0}_name'.format(s)] = '{0}:{1}'.format(s, name)

            val = options['kwargs'].get('val', np.ones(1))
            if isinstance(val, Number):
                n = 1
            else:
                n = len(val)
            self._output_vars[name]['size'] = n

            self.add_output(name, **options['kwargs'])

            self.add_input(options['lhs_name'],
                           val=np.ones(n),
                           units=options['eq_units'])

            self.add_input(options['rhs_name'],
                           val=options['rhs_val'] * np.ones(n),
                           units=options['eq_units'])

            if options['use_mult']:
                self.add_input(options['mult_name'],
                               val=options['mult_val'] * np.ones(n),
                               units=None)

            self._scale_factor = np.ones(n)
            self._dscale_drhs = np.ones(n)

            ar = np.arange(n)
            self.declare_partials(of=name, wrt=options['lhs_name'], rows=ar, cols=ar, val=1.0)
            self.declare_partials(of=name, wrt=options['rhs_name'], rows=ar, cols=ar, val=1.0)

            if options['use_mult']:
                self.declare_partials(of=name, wrt=options['mult_name'], rows=ar, cols=ar, val=1.0)

            if options['add_constraint']:
                self.add_constraint(name, equals=0.)

    def compute(self, inputs, outputs):
        """
        Calculate the output for each equality constraint.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        for name, options in iteritems(self._output_vars):
            lhs = inputs[options['lhs_name']]
            rhs = inputs[options['rhs_name']]

            # Indices where the rhs is near zero or not near zero
            idxs_nz = np.where(np.abs(rhs) < 2)[0]
            idxs_nnz = np.where(np.abs(rhs) >= 2)[0]

            # Compute scaling factors
            # scale factor that normalizes by the rhs, except near 0
            self._scale_factor[idxs_nnz] = 1.0 / np.abs(rhs[idxs_nnz])
            self._scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

            if options['use_mult']:
                outputs[name] = (inputs[options['mult_name']] * lhs - rhs) * self._scale_factor
            else:
                outputs[name] = (lhs - rhs) * self._scale_factor

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        for name, options in iteritems(self._output_vars):
            lhs_name = options['lhs_name']
            rhs_name = options['rhs_name']

            lhs = inputs[lhs_name]
            rhs = inputs[rhs_name]

            # Indices where the rhs is near zero or not near zero
            idxs_nz = np.where(np.abs(rhs) < 2)[0]
            idxs_nnz = np.where(np.abs(rhs) >= 2)[0]

            # scale factor that normalizes by the rhs, except near 0
            self._scale_factor[idxs_nnz] = 1.0 / np.abs(rhs[idxs_nnz])
            self._scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

            self._dscale_drhs[idxs_nnz] = -np.sign(rhs[idxs_nnz]) / rhs[idxs_nnz]**2
            self._dscale_drhs[idxs_nz] = -.5 * rhs[idxs_nz] / (.25 * rhs[idxs_nz] ** 2 + 1) ** 2

            if options['use_mult']:
                mult_name = options['mult_name']
                mult = inputs[mult_name]

                # Partials of output wrt mult
                partials[name, mult_name] = lhs * self._scale_factor
            else:
                mult = 1.0

            # Partials of output wrt rhs
            partials[name, rhs_name] = (mult * lhs - rhs) * self._dscale_drhs - self._scale_factor

            # Partials of output wrt lhs
            partials[name, lhs_name] = mult * self._scale_factor

    def add_eq_output(self, name, eq_units=None, lhs_name=None, rhs_name=None, rhs_val=0.0,
                      use_mult=False, mult_name=None, mult_val=1.0, add_constraint=False,
                      **kwargs):
        """
        Add a new output variable computed via the difference equation.

        This will create new inputs `lhs:name`, `rhs:name`, and `mult:name` that will
        define the left and right sides of the difference equation, and a
        multiplier for the left-hand-side.

        Parameters
        ----------
        name : str
            The name of the output variable to be created.
        eq_units : str or None
            Units for the left-hand-side and right-hand-side of the difference equation.
        lhs_name : str or None
            Optional name for the LHS variable associated with the difference equation.  If
            None, the default will be used:  'lhs:{name}'.
        rhs_name : str or None
            Optional name for the RHS variable associated with the difference equation.  If
            None, the default will be used:  'rhs:{name}'.
        rhs_val : int, float, or np.array
            Default value for the RHS.  Must be compatible with the shape (optionally)
            given by the val or shape option in kwargs.
        use_mult : bool
            Specifies whether the LHS multiplier is to be used.  If True, then an additional
            input `mult_name` is created, with the default value given by `mult_val`, that
            multiplies lhs.  Default is False.
        mult_name : str or None
            Optional name for the LHS multiplier variable associated with the output
            variable. If None, the default will be used: 'mult:{name}'.
        mult_val : int, float, or np.array
            Default value for the LHS multiplier.  Must be compatible with the shape (optionally)
            given by the val or shape option in kwargs.
        add_constraint : bool
            Specifies whether to add an equality constraint.
        **kwargs : dict
            Additional arguments to be passed for the creation of the output variable.
            (see `add_output` method).
        """
        self._output_vars[name] = {'kwargs': kwargs,
                                   'eq_units': eq_units,
                                   'lhs_name': lhs_name,
                                   'rhs_name': rhs_name,
                                   'rhs_val': rhs_val,
                                   'use_mult': use_mult,
                                   'mult_name': mult_name,
                                   'mult_val': mult_val,
                                   'add_constraint': add_constraint}
