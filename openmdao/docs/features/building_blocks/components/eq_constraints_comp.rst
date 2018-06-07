.. index:: EqualityConstraintsComp Example

.. _eq_constraints_comp_feature:

***********************
EqualityConstraintsComp
***********************

`EqualityConstraintsComp` is a specialized component that provides a simple way to implement
equality constraints.

You can add one or more outputs to an `EqualityConstraintsComp` that compute
the difference between a pair of input values for the purposes of driving the two inputs to equality.

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

The `EqualityConstraintsComp` supports vectorized outputs. Simply provide a default
value or shape when adding the difference equation that reflects the correct shape.

`EqualityConstraintsComp` accepts the following other arguments (which are all passed
to ``add_eq_output`` during initialization):

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

Example: Sellar IDF
-------------------

The following example shows an Individual Design Feasible (IDF) architecture for the
:ref:`Sellar <sellar>` problem that demonstrates the use of an `EqualityConstraintsComp`.

In IDF, the direct coupling between the disciplines is removed and the coupling variables
are added to the optimizer’s design variables. The algorithm calls for two new equality
constraints that enforce the coupling between the disciplines. This ensures that the
solution is a feasible coupling, though it is achieved through the optimizer instead of
using a solver.  The two new equality constraints are implemented in this example with
an `EqualityConstraintsComp`.

.. embed-code::
    openmdao.components.tests.test_eq_constraints_comp.SellarIDF
    :layout: code

.. embed-code::
    openmdao.components.tests.test_eq_constraints_comp.TestEqualityConstraintsComp.test_feature_sellar_idf
    :layout: interleave

.. tags:: EqualityConstraintsComp, Component
