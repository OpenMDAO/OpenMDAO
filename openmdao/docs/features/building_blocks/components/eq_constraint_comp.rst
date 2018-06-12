.. index:: EQConstraintComp Example

.. _eq_constraint_comp_feature:

***********************
EQConstraintComp
***********************

`EQConstraintComp` is a specialized component that provides a simple way to implement
equality constraints.

You can add one or more outputs to an `EQConstraintComp` that compute the difference
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

The default value for the :code:`rhs:{name}` input can be set to via the
:code:`rhs_val` argument (see arguments below). If the rhs value is fixed (e.g. 0),
then the input can be left unconnected. The :code:`lhs:{name}` input must always have
something connected to it.

The multiplier is optional and will default to 1.0 if not connected.

`EQConstraintComp` supports vectorized outputs. Simply provide a default
value or shape when adding the output that reflects the correct shape.

You can provide the arguments to create an output variable when instantiating an
`EQConstraintComp` or you can use the ``add_eq_output`` method to create one
or more outputs after instantiation.  The constructor accepts all the same arguments
as the ``add_eq_output`` method:

.. automethod:: openmdao.components.eq_constraint_comp.EQConstraintComp.add_eq_output
   :noindex:

Note that the `kwargs` arguments can include any of the keyword arguments normally available
when creating an output variable with the
:meth:`add_output <openmdao.core.component.Component.add_output>` method of a `Component`.


Example: Sellar IDF
-------------------

The following example shows an Individual Design Feasible (IDF) architecture for the
:ref:`Sellar <sellar>` problem that demonstrates the use of an `EQConstraintComp`.

In IDF, the direct coupling between the disciplines is removed and the coupling variables
are added to the optimizer’s design variables. The algorithm calls for two new equality
constraints that enforce the coupling between the disciplines. This ensures that the final
optimized solution is feasible, though it is achieved through the optimizer instead of
using a solver.  The two new equality constraints are implemented in this example with
an `EQConstraintComp`.

.. embed-code::
    openmdao.components.tests.test_eq_constraint_comp.SellarIDF
    :layout: code

.. embed-code::
    openmdao.components.tests.test_eq_constraint_comp.TestFeatureEQConstraintComp.test_feature_sellar_idf
    :layout: interleave

.. tags:: EQConstraintComp, Component
