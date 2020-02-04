.. index:: BsplinesComp Example

.. _bsplinescomp_feature:

************
BsplinesComp
************

The BsplinesComp allows you to represent a large dimension variable as a smaller dimensional variable via a B-spline function.
This is useful for reducing the size of an optimization problem by decreasing the number of design variables it solves, and potentially
the number of gradients it computes if you are running in forward mode. You may consider using this when you have spatially
distributed design variables where you know that the physics gives you smooth designs that can be well approximated by B-splines.

BsplinesComp Options
--------------------

.. embed-options::
    openmdao.components.bsplines_comp
    BsplinesComp
    options


BsplinesComp Example
--------------------

The following is a simple example where we use a BsplineComp to take a 10 point sine wave spanning 90
degrees and represent it with 5 control points that can be used as design variables.

.. embed-code::
    openmdao.components.tests.test_bsplines_comp.TestBsplinesCompFeature.test_basic
    :layout: code, output

The BsplinesComp is vectorized, so you can operate on a multi-row vector and have the it create
interpolated points independently for each row:

.. embed-code::
    openmdao.components.tests.test_bsplines_comp.TestBsplinesCompFeature.test_vectorized
    :layout: code, output


BsplinesComp Option Examples
----------------------------

**distribution**

The :code:`distribution` option can be used to change the spacing of the spline control points.
A "uniform" distribution yields a set of evenly distributed control points, while a "sine"
distribution places more points towards the edges.

For example, let's say we have a spatial distributed variable, like the beam thickness
in the :ref:`beam optimization <beam_optimization_example_part_2>` example, that has 100 nodes.
We would like to reduce that to a more reasonable number like 20, so we use the BsplineComp.
Our initial value for this variable is roughly a sine wave. When we create the BsplineComp
with a "uniform" distribution, our control points are evenly spaced over the domain, as seen
in the figure below.

.. embed-code::
    openmdao.components.tests.test_bsplines_comp.TestBsplinesCompFeatureWithPlotting.test_distribution_uniform
    :layout: code, plot
    :scale: 90
    :align: center

However, when we choose "sine" for the distribution, we end up with more control points towards the two edges
as seen below. This is beneficial if we know that the optimal design will have more variation (or rather,
higher spatial frequency content) near the edges than the middle.

.. embed-code::
    openmdao.components.tests.test_bsplines_comp.TestBsplinesCompFeatureWithPlotting.test_distribution_sine
    :layout: code, plot
    :scale: 90
    :align: center

.. tags:: BsplinesComp, Component
