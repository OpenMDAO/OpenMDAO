"""Define the EQConstraintComp class."""

from numbers import Number

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils import cs_safe
from openmdao.utils.array_utils import shape_to_len


class EQConstraintComp(ExplicitComponent):
    """
    A component that computes the difference between two inputs to test for equality.

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
    normalize : bool
        Specifies whether or not the resulting output should be normalized by the RHS.  When
        the RHS value is between [-2, 2], the normalization value is a quadratic function that
        is close to one but still provides a C1 continuous function. When this option is True,
        the user-provided ref/ref0 scaler/adder options below are typically unnecessary.
    add_constraint : bool
        Specifies whether to add an equality constraint.
    ref : float or ndarray, optional
        Value of response variable that scales to 1.0 in the driver. This option is only
        meaningful when add_constraint=True.
    ref0 : float or ndarray, optional
        Value of response variable that scales to 0.0 in the driver. This option is only
        meaningful when add_constraint=True.
    adder : float or ndarray, optional
        Value to add to the model value to get the scaled value for the driver. adder
        is first in precedence. This option is only meaningful when add_constraint=True.
    scaler : float or ndarray, optional
        Value to multiply the model value to get the scaled value for the driver. scaler
        is second in precedence. This option is only meaningful when add_constraint=True.
    **kwargs : dict
        Additional arguments to be passed for the creation of the output variable.
        (see `add_output` method).

    Attributes
    ----------
    _output_vars : dict
        Cache the data provided during `add_eq_output`
        so everything can be saved until setup is called.
    """

    def __init__(self, name=None, eq_units=None, lhs_name=None, rhs_name=None, rhs_val=0.0,
                 use_mult=False, mult_name=None, mult_val=1.0, normalize=True, add_constraint=False,
                 ref=None, ref0=None, adder=None, scaler=None, **kwargs):
        r"""
        Initialize an EQConstraintComp, optionally add an output constraint to the model.

        The EQConstraintComp allows for the creation of one or more output variables and
        computes the values for those variables based on the following equation:

        .. math::

          name_{output} = \frac{name_{mult} \times name_{lhs} - name_{rhs} }{f_{norm}(name_{rhs})}

        Where :math:`name_{lhs}` represents the left-hand-side of the equality,
        :math:`name_{rhs}` represents the right-hand-side, and :math:`name_{mult}`
        is an optional multiplier on the left hand side. If use_mult is True then
        the default value of the multiplier is 1.  The optional normalization function
        :math:`f_{norm}` is computed as:

        .. math::

          f_{norm}(name_{rhs}) =
          \begin{cases}
           \left| name_{rhs} \right|, & \text{if normalize and } \left| name_{rhs} \right| \geq 2 \\
           0.25 name_{rhs}^2 + 1,     & \text{if normalize and } \left| name_{rhs} \right| < 2 \\
           1,                         & \text{if not normalize}
          \end{cases}

        New output variables are created by calling `add_eq_output`.
        """
        super().__init__()
        self._output_vars = {}
        if name is not None:
            self.add_eq_output(name, eq_units, lhs_name, rhs_name, rhs_val,
                               use_mult, mult_name, mult_val, normalize, add_constraint, ref, ref0,
                               adder, scaler, **kwargs)

        self._no_check_partials = True

    def compute(self, inputs, outputs):
        """
        Calculate the output for each equality constraint.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        for name, options in self._output_vars.items():
            lhs = inputs[options['lhs_name']]
            rhs = inputs[options['rhs_name']]

            # set dtype to rhs.dtype to prevent
            # "Casting complex values to real discards the imaginary part" warning.
            _scale_factor = np.ones((rhs.shape), dtype=rhs.dtype)

            # Compute scaling factors
            # scale factor that normalizes by the rhs, except near 0
            if options['normalize']:
                # Indices where the rhs is near zero or not near zero
                idxs_nz = np.where(cs_safe.abs(rhs) < 2)
                idxs_nnz = np.where(cs_safe.abs(rhs) >= 2)

                _scale_factor[idxs_nnz] = 1.0 / cs_safe.abs(rhs[idxs_nnz])
                _scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

            if options['use_mult']:
                outputs[name] = (inputs[options['mult_name']] * lhs - rhs) * _scale_factor
            else:
                outputs[name] = (lhs - rhs) * _scale_factor

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
        for name, options in self._output_vars.items():
            lhs_name = options['lhs_name']
            rhs_name = options['rhs_name']

            lhs = inputs[lhs_name]
            rhs = inputs[rhs_name]

            _scale_factor = np.ones((rhs.shape))
            _dscale_drhs = np.zeros((rhs.shape))
            if options['normalize']:
                # Indices where the rhs is near zero or not near zero
                idxs_nz = np.where(cs_safe.abs(rhs) < 2)
                idxs_nnz = np.where(cs_safe.abs(rhs) >= 2)

                # scale factor that normalizes by the rhs, except near 0
                _scale_factor[idxs_nnz] = 1.0 / cs_safe.abs(rhs[idxs_nnz])
                _scale_factor[idxs_nz] = 1.0 / (.25 * rhs[idxs_nz] ** 2 + 1)

                _dscale_drhs[idxs_nnz] = -np.sign(rhs[idxs_nnz]) / rhs[idxs_nnz]**2
                _dscale_drhs[idxs_nz] = -.5 * rhs[idxs_nz] / (.25 * rhs[idxs_nz] ** 2 + 1) ** 2

            if options['use_mult']:
                mult_name = options['mult_name']
                mult = inputs[mult_name]

                # Partials of output wrt mult
                deriv = lhs * _scale_factor
                partials[name, mult_name] = deriv.flatten()
            else:
                mult = 1.0

            # Partials of output wrt rhs
            deriv = (mult * lhs - rhs) * _dscale_drhs - _scale_factor
            partials[name, rhs_name] = deriv.flatten()

            # Partials of output wrt lhs
            deriv = mult * _scale_factor
            partials[name, lhs_name] = deriv.flatten()

    def add_eq_output(self, name, eq_units=None, lhs_name=None, rhs_name=None, rhs_val=0.0,
                      use_mult=False, mult_name=None, mult_val=1.0, normalize=True,
                      add_constraint=False, ref=None, ref0=None, adder=None, scaler=None,
                      linear=False, indices=None, cache_linear_solution=False,
                      flat_indices=False, alias=None, **kwargs):
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
        normalize : bool
            Specifies whether or not the resulting output should be normalized by a quadratic
            function of the RHS. When this option is True, the user-provided ref/ref0 scaler/adder
            options below are typically unnecessary.
        add_constraint : bool
            Specifies whether to add an equality constraint.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver. This option is only
            meaningful when add_constraint=True.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver. This option is only
            meaningful when add_constraint=True.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence. This option is only meaningful when add_constraint=True.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. This option is only meaningful when add_constraint=True.
        linear : bool
            Set to True if constraint is linear. Default is False.
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.  These may be positive or
            negative integers.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        flat_indices : bool
            If True, interpret specified indices as being indices into a flat source array.
        alias : str
            Alias for this response. Necessary when adding multiple constraints on different
            indices or slices of a single variable.
        **kwargs : dict
            Additional arguments to be passed for the creation of the output variable.
            (see `add_output` method).
        """
        self._output_vars[name] = options = {'kwargs': kwargs,
                                             'eq_units': eq_units,
                                             'lhs_name': lhs_name,
                                             'rhs_name': rhs_name,
                                             'rhs_val': rhs_val,
                                             'use_mult': use_mult,
                                             'mult_name': mult_name,
                                             'mult_val': mult_val,
                                             'normalize': normalize,
                                             'add_constraint': add_constraint,
                                             'ref': ref,
                                             'ref0': ref0,
                                             'adder': adder,
                                             'scaler': scaler}

        meta = self.add_output(name, **options['kwargs'])

        shape = meta['shape']

        for s in ('lhs', 'rhs', 'mult'):
            if options[f'{s}_name'] is None:
                options[f'{s}_name'] = f'{s}:{name}'

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

        ar = np.arange(shape_to_len(shape))
        self.declare_partials(of=name, wrt=options['lhs_name'], rows=ar, cols=ar, val=1.0)
        self.declare_partials(of=name, wrt=options['rhs_name'], rows=ar, cols=ar, val=1.0)

        if options['use_mult']:
            self.declare_partials(of=name, wrt=options['mult_name'], rows=ar, cols=ar, val=1.0)

        if options['add_constraint']:
            self.add_constraint(name, equals=0., ref0=options['ref0'], ref=options['ref'],
                                adder=options['adder'], scaler=options['scaler'],
                                linear=linear, indices=indices, flat_indices=flat_indices,
                                cache_linear_solution=cache_linear_solution, alias=alias)
