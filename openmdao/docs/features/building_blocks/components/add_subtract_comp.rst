
.. _addsubtractcomp_feature:

********************************************
AddSubtractComp
********************************************

:ref:`AddSubtractComp <openmdao.components.add_subtract_comp.py>` performs elementwise addition or subtraction between two or more compatible inputs.  It may be vectorized to provide the result at one or more points simultaneously.

.. math::

    result = a * \textrm{scaling factor}_a + b * \textrm{scaling factor}_b + c * \textrm{scaling factor}_c + ...

Using the AddSubtractComp
---------------------------------------------------

The `add_equation` method is used to set up a system of inputs to be added/subtracted (with scaling factors).
Each time the user adds an equation, all of the inputs and outputs must be of identical shape (this is a requirement for elementwise addition/subtraction).
The units must also be compatible between all inputs and the output of each equation.

AddSubtractComp Example
---------------------------------------------------

In the following example AddSubtractComp is used to add thrust, drag, lift, and weight forces. Note the scaling factor of -1 for the drag force and weight force.

.. embed-code::
    openmdao.components.tests.test_add_subtract_comp.TestFeature.test
    :layout: code, output

.. tags:: AddSubtractComp, Component
