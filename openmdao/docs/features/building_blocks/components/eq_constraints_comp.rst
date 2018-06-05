.. index:: EqualityConstraintsComp Example

.. _eq_constraints_comp_feature:

***********************
EqualityConstraintsComp
***********************

`EqualityConstraintsComp` is a specialized component that provides a simple way to implement
equality constraints.

`EqualityConstraintsComp` allows you to add one or more outputs that compute the difference
between a pair of input values for the purposes of driving the two inputs to equality.

The following inputs and outputs are associated with each output variable.

=========== ======= ====================================================
Name        I/O     Description
=========== ======= ====================================================
{name}      output  output variable
lhs:{name}  input   left-hand side of difference equation
rhs:{name}  input   right-hand side of difference equation
mult:{name} input   left-hand side multiplier of difference equation
=========== ======= ====================================================

The right-hand side is optional and will default to zero if not connected.
The multiplier is optional and will default to 1.0 if not connected. The
left-hand side should always be defined.

The `EqualityConstraintsComp` supports vectorized outputs, simply provide a default
value or shape when adding the difference equation that reflects the correct shape.

`EqualityConstraintsComp` accepts the following other arguments (which are all passed
to ``add_eq_output equation`` during initialization):

=============== ======================== ===================================================================================
Name            Type                     Description
=============== ======================== ===================================================================================
eq_units        str or None              Units associated with left-hand and right-hand side. (mult is treated as unitless).
lhs_name        str or None              Optional name associated with the left-hand side of the difference equation.
rhs_name        str or None              Optional name associated with the right-hand side of the difference equation.
rhs_val         int, float, or np.array  Default value for the right-hand side.
use_mult        bool                     Specifies whether the left-hand side multiplier is to be used.
mult_name       str or None              Optional name associated with the left-hand side multiplier variable.
mult_val        int, float, or np.array  Default value for the left-hand side multiplier.
add_constraint  bool                     Specifies whether to add an equality constraint.
kwargs          dict or named arguments  Additional arguments to be passed for the creation of the output variable.
=============== ======================== ===================================================================================

Simple Example
--------------

The following simple example uses an EqualityConstraintsComp to find the intersection of two parabolas.
The equations for the two parabolas are:

.. math::

    f(x) = 1.5^2 - 9x + 11.5

.. math::
    g(x) = -0.2^2 - 0.4x + 2.8

and they look like this:

.. embed-code::
    openmdao.components.tests.test_eq_constraints_comp.TestEqualityConstraintsComp.plot_feature_two_parabolas
    :layout: plot
    :scale: 90
    :align: center

Here we use the `EqualityConstraintsComp` to constrain the :code:`y` values to be equal while we optimize for the
minimum value:

.. embed-code::
    openmdao.components.tests.test_eq_constraints_comp.TestEqualityConstraintsComp.test_feature_two_parabolas
    :layout: interleave

.. tags:: EqualityConstraintsComp, Component
