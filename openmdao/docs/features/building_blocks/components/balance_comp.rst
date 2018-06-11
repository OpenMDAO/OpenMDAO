.. index:: BalanceComp Example

.. _balancecomp_feature:

***********
BalanceComp
***********

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

The BalanceComp supports vectorized implicit states. Simply provide a default
value or shape when adding the balance that reflects the correct shape.

You can provide the arguments to create a balance when instantiating a `BalanceComp`
or you can use the ``add_balance`` method to create one or more state variables after
instantiation.  The constructor accepts all the same arguments as the ``add_balance``
method:

.. automethod:: openmdao.components.balance_comp.BalanceComp.add_balance
   :noindex:

Note that the `kwargs` arguments can include any of the keyword arguments normally available
when creating an output variable with the
:meth:`add_output <openmdao.core.component.Component.add_output>` method of a `Component`.


Example:  Scalar Root Finding
-----------------------------

The following example uses a BalanceComp to implicitly solve the
equation:

.. math::

    2 \cdot x^2 = 4

Here, our LHS is connected to a computed value for :math:`x^2`, the multiplier is 2, and the RHS
is 4.  The expected solution is :math:`x=\sqrt{2}`.  We initialize :math:`x` with a value of 1 so that
it finds the positive root.

.. embed-code::
    openmdao.components.tests.test_balance_comp.TestBalanceComp.test_feature_scalar
    :layout: interleave

Alternatively, we could simplify the code by using the :code:`mult_val` argument.

.. embed-code::
    openmdao.components.tests.test_balance_comp.TestBalanceComp.test_feature_scalar_with_default_mult
    :layout: interleave


Example:  Vectorized Root Finding
---------------------------------

The following example uses a BalanceComp to implicitly solve the equation:

.. math::

    b \cdot x + c  = 0

for various values of :math:`b`, and :math:`c`.  Here, our LHS is connected to a computed value of
the linear equation.  The multiplier is one and the RHS is zero (the defaults), and thus
they need not be connected.

.. embed-code::
    openmdao.components.tests.test_balance_comp.TestBalanceComp.test_feature_vector
    :layout: interleave


Example:  Providing an Initial Guess for a State Variable
---------------------------------------------------------

As mentioned above, there is an optional argument to :code:`add_balance` called :code:`guess_func` which can
provide an initial guess for a state variable.

The Kepler example script shows how :code:`guess_func` can be used.

.. embed-code::
    openmdao.test_suite.test_examples.test_keplers_equation.TestKeplersEquation.test_result
    :layout: interleave

.. tags:: BalanceComp, Component
