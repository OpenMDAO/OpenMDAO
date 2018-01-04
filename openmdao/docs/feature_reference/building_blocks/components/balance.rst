.. index:: BalanceComp Example

.. _balancecomp_feature:

*****************
Balance Component
*****************

`BalanceComp` is a specialized implementation of `ImplicitComponent` that
is intended to provide a simple way to implement most implicit equations
without the need to define your own residuals.

`BalanceComp` allows you to add one or more state variables and its associated
implicit equations.  For each ``balance`` added to the component it
solves the following equation:

.. math::

    f_{mult}(x) \cdot f_{lhs}(x) = f_{rhs}(x)

The following inputs and outputs are associated with each implicit state.

=========== ======= ====================================================
Name        I/O     Description
=========== ======= ====================================================
{name}      output  implicit state variable
lhs:{name}  input   left-hand side of equation to be balanced
rhs:{name}  input   right-hand side of equation to be balanced
mult:{name} input   left-hand side multiplier of equation to be balanced
=========== ======= ====================================================

The right-hand side is optional and will default to zero if not connected.
The multiplier is optional and will default to 1.0 if not connected. The
left-hand side should always be defined and should be dependent upon the value
of the implicit state variable.

The BalanceComp supports vectorized implicit states, simply provide a default
value or shape when adding the balance that reflects the correct shape.

Balance accepts the following other arguments (which are all passed
to ``add_balance`` during initialization):

=========== ======================== ==================================================================================
Name        Type                     Description
=========== ======================== ==================================================================================
eq_units    str or None              Units associated with LHS and RHS.  (mult is treated as unitless)
lhs_name    str or None              Optional name associated with the left-hand side of the balance.
rhs_name    str or None              Optional name associated with the right-hand side of the balance.
mult_name   str or None              Optional name associated with the right-hand side of the balance.
rhs_val     int, float, or np.array  Default value for the RHS.
mult_val    int, float, or np.array  Default value for the multiplier.
kwargs      dict or named arguments  Additional arguments to be passed for the creation of the implicit state variable.
=========== ======================== ==================================================================================

Example:  Scalar Root Finding
-----------------------------

The following example uses the Balance Component to implicitly solve the
equation:

.. math::

    2 \cdot x^2 = 4

Here, our LHS is connected to a computed value for :math:`x^2`, the multiplier is 2, and the RHS
is 4.  The expected solution is :math:`x=\sqrt{2}`.  We initialize ``x`` with a value of 1 so that
it finds the positive root.

.. embed-test::
    openmdao.components.tests.test_balance_comp.TestBalanceComp.test_feature_scalar

Example:  Vectorized Root Finding
---------------------------------

The following example uses the Balance Component to implicitly solve the
equation:

.. math::

    b \cdot x + c  = 0

for various values of ``b``, and ``c``.  Here, our LHS is connected to a computed value of
the linear equation.  The multiplier is one and the RHS is zero (the defaults) and thus
they need not be connected.

.. embed-test::
    openmdao.components.tests.test_balance_comp.TestBalanceComp.test_feature_vector

.. tags:: BalanceComp
