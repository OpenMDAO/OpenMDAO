.. index:: KSComponent Example

.. _kscomponent_feature:

***********
KSComponent
***********

KSComponent provides a way to aggregate many constraints into a single constraint. This is usually done for performance
reasons, in particular, to reduce the calculation time needed for the total derivatives of your model. The KSComponent
implements the Kreisselmeier-Steinhauser Function to aggregate constraint vector input "g" into a single scalar output 'KS'.

By default, the constraint vector "g" is assumed be of the form where g<=0 satisfies the constraints, but other forms can
be specified using the "upper" and "lower_flag" options.

KSComponent Options
-------------------

.. embed-options::
    openmdao.components.ks
    KSComponent
    options


KSComponent Example
-------------------

The following example is perhaps the simplest possible. It shows a component that represents a constraint
of width two. We would like to aggregate the values of this constraint vector into a single scalar
value using the KSComponent.

.. embed-code::
    openmdao.components.tests.test_ks.TestKSFunctionFeatures.test_basic

A more practical example that uses the KSComponent can be found in the :ref:`beam optimization <beam_optimization_example_part_2>` example.

KSComponent Option Examples
---------------------------

Normally, the input constraint vector is assumed to be of the form g<=0 is satisfied. If you would like to set a
different upper bound for the constraint, you can declare it in the "upper" option in the options dictionary.

In the following example, we specify a new upper bound of 16 for the constraint vector. Note that the KS output
is still satisfied if it is less than zero.

**upper**

.. embed-code::
    openmdao.components.tests.test_ks.TestKSFunctionFeatures.test_upper
    :layout: interleave

Normally, the input constraint vector is satisfied if it is negative and violated if it is positive. You can
reverse this behavior by setting the "lower_flag" option to True. In the following example, we turn on the
"lower_flag" so that positive values of the input constraint are considered satisfied. Note that the KS output
is still satisfied if it is less than zero.

**lower_flag**

.. embed-code::
    openmdao.components.tests.test_ks.TestKSFunctionFeatures.test_lower_flag
    :layout: interleave


.. tags:: KSComponent, Constraints, Optimization