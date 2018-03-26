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

The following example shows the optimization of a simple beam with rectangular cross section. The beam has been subdivided into
elements along the length of the beam, and one end is fixed. The goal is to minimize the volume (and hence the mass of the
homogeneous beam) by varying the thickness in each element without exceeding a maximum stress constraint while the beam is
subject to multiple load cases, each one being a distributed force load that varies sinusoidally along the span.

Constraining the bending stress on each element leads to a more computationally expensive derivative calculation, so we
will use the KSFunction to reduce the stress vector for each load case to a single scalar value. To do so, we also need
to insert an `ExecComp` component that converts the stress into a form where a negative value means it is satisfied, and
a positive value means it is violated.

The problem presented here is also an example of a multi-point implementation, where we create a separate instance of the
parts of the calculation that are impacted by different loadcases. This enables our model to take advantage of multiple
processors when run in parallel.

The code for the top system is this:

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress

Next we run the model, and choose `ScipyOptimizeDriver` SLSQP to be our optimizer. At the conclusion of optimization,
we print out the design variable, which is the thickness for each element.

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.test_beam_optimization.TestCase.test_multipoint_stress
    :layout: interleave


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