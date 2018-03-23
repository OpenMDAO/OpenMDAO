.. index:: BsplinesComp Example

.. _bsplinescomp_feature:

************
BsplinesComp
************

The BsplinesComp allows you to represent a large dimension variable as a smaller dimensional variable via a B-spline function.
This is useful for reducing the size of an optimization problem by decreasing the number of design variables it solves, and potentially
the number of gradients it computes if you are running in forward mode. You may consider using this when you have spatially
distributed design variables where you know that the physics gives you smooth designs that can be well approximated by B-splines.


BsplinesComp Example
--------------------

The following example shows the optimization of a simple beam with rectangular cross section. The beam has been subdivided into
elements along the length of the beam, and one end is fixed. The goal is to minimize the volume (and hence the mass of the
homogeneous beam) by varying the thickness in each element without exceeding a maximum stress constraint while the beam is
subject to multiple load cases, each one being a distributed force load that varies sinusoidally along the span.

If we allow the optimizer to vary the thickness of each element, then we have a design variable vector that is as wide as the
number of elements in the model. This may perform poorly if we have a large number of elements in the beam. If we assume that
the optimal beam thickness is going to have a smooth continuous variation over the length of the beam, then it is a good
candidate for using an interpolation component like BsplinesComp.

For this example, we have 25 elements, but can reduce that to 5 control points for the optimizer's design variables by
including the BsplinesComp.


The code for the top system is this:

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress

There are 4 arguments to the BsplineComp that are required: the number of control points, the number of interpolation points,
the input variable name, and the output variable name.

Next we run the model, and choose `ScipyOptimizeDriver` SLSQP to be our optimizer. At the conclusion of optimization,
we print out the design variable, which is the thickness for each element.

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.test_beam_optimization.TestCase.test_multipoint_stress
    :layout: interleave


BsplinesComp Option Examples
----------------------------

**distribution**

The "distribution" option can be used to change the spacing of the spline control points. A uniform distribution
yields a set of evenly distributed control points, while a "sine" distribution places more points towards the edges.

For example, let's say we have a spatial distributed variable, like the beam thickness in the previous example, that
has 100 nodes. We would like to reduce that to a more reasonable number like 20, so we use the BsplineComp. Our
initial value for this variable is roughly a sine wave. When we create the BsplineComp and with a "uniform"
distribution, our control points are evenly spaced over the domain, as seen in the figure below.

.. embed-code::
    openmdao.components.tests.test_interp.TestBsplinesCompFeature.test_distribution_uniform
    :layout: interleave, plot
    :scale: 90
    :align: center

However, when we choose "sine" for the distribution, we end up with more control points towards the two edges
as seen below. This is beneficial if we know that the optimal design will have more variation (or rather,
higher spatial frequency content) near the edges than the middle.

.. embed-code::
    openmdao.components.tests.test_interp.TestBsplinesCompFeature.test_distribution_sine
    :layout: interleave, plot
    :scale: 90
    :align: center