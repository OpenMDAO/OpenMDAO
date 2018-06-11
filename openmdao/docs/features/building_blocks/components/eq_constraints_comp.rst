.. index:: EqualityConstraintsComp Example

.. _eq_constraints_comp_feature:

***********************
EqualityConstraintsComp
***********************

`EqualityConstraintsComp` is a specialized component that provides a simple way to implement
equality constraints.

You can add one or more outputs to an `EqualityConstraintsComp` that compute the difference
between a pair of input values for the purposes of driving the two inputs to equality. It
computes the output value as:

.. math::

  name_{output} = name_{mult} \times name_{lhs} - name_{rhs}


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
value or shape when adding the output that reflects the correct shape.

You can provide the arguments to create an output variable when instantiating an
`EqualityConstraintsComp` or you can use the ``add_eq_output`` method to create one
or more outputs after instantiation.  The constructor accepts all the same arguments
as the ``add_eq_output`` method:

.. automethod:: openmdao.components.eq_constraints_comp.EqualityConstraintsComp.add_eq_output
   :noindex:

Note that the `kwargs` arguments can include any of the keyword arguments normally available
when creatiing of an output variable using the
:meth:`add_output <openmdao.core.component.Component.add_output>` method of a `Component`.

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
    openmdao.components.tests.test_eq_constraints_comp.TestFeatureEqualityConstraintsComp.test_feature_sellar_idf
    :layout: interleave

.. tags:: EqualityConstraintsComp, Component
