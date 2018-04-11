.. _beam_optimization_example_part_2:

Revisiting the Beam Problem - Minimizing Stress with KS Constraints and BSplines
================================================================================

The following example shows the optimization of a simple beam with rectangular cross section. The beam has been subdivided into
elements along the length of the beam, and one end is fixed. The goal is to minimize the volume (and hence the mass of the
homogeneous beam) by varying the thickness in each element without exceeding a maximum stress constraint while the beam is
subject to multiple load cases, each one being a distributed force load that varies sinusoidally along the span.

Constraining the bending stress on each element leads to a more computationally expensive derivative calculation, so we
will use the `KSFunction` to reduce the stress vector for each load case to a single scalar value. To do so, we also need
to insert an `ExecComp` component that converts the stress into a form where a negative value means it is satisfied, and
a positive value means it is violated.

The problem presented here is also an example of a multi-point implementation, where we create a separate instance of the
parts of the calculation that are impacted by different loadcases. This enables our model to take advantage of multiple
processors when run in parallel.

If we allow the optimizer to vary the thickness of each element, then we have a design variable vector that is as wide as the
number of elements in the model. This may perform poorly if we have a large number of elements in the beam. If we assume that
the optimal beam thickness is going to have a smooth continuous variation over the length of the beam, then it is a good
candidate for using an interpolation component like `BsplinesComp` to reduce the number of design variables we need.

For this example, we have 25 elements, but can reduce that to 5 control points for the optimizer's design variables by
including the BsplinesComp.

The code for the top system is this:

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress

Next we run the model, and choose `ScipyOptimizeDriver` SLSQP to be our optimizer. At the conclusion of optimization,
we print out the design variable, which is the thickness for each element.

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.test_beam_optimization.TestCase.test_multipoint_stress
    :layout: interleave