.. index:: BsplinesComp Example

.. _bsplinescomp_feature:

************
BsplinesComp
************


BsplinesComp Example
--------------------

The following example shows the optimization of a simple beam with rectangular cross section. The beam has been subdivided into
elements along the length of the beam, and one end is fixed. The goal is to minimize the volume (and hence the mass of the
homogeneous beam) by varying the thickness in each element without exceeding a maximum stress constraint while the beam is
subject to multiple load cases, each one being a distributed force load that varies sinusoidally along the span.

If we allow the optimizer to vary the thickness of each element, then we have a design variable vector that is as wide as the
number of elements in the model. This may perform poorly if we have a large number of elements in the beam. If we assume that
the optimal beam thickness is going to have a smooth continuous variation over the length of the beam, then it is a good
candidate for using an interpolation component like BSplinesComp.

For this example, we have 25 elements, but can reduce that to 5 control points for the optimizer's design variables by
including the BSplinesComp.


The code for the top system is this:

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress

Next we run the model, and choose `ScipyOptimizeDriver` SLSQP to be our optimizer. At the conclusion of optimization,
we print out the design variable, which is the thickness for each element.

.. embed-code::
    openmdao.test_suite.test_examples.beam_optimization.test_beam_optimization.TestCase.test_multipoint_stress
    :layout: interleave


BsplinesComp Option Examples
----------------------------

**distribution**

.. embed-code::
    openmdao.components.tests.test_interp.TestBSplinesCompFeature.test_distribution_uniform
    :layout: interleave, plot
    :align: center

.. embed-code::
    openmdao.components.tests.test_interp.TestBSplinesCompFeature.test_distribution_sine
    :layout: interleave, plot
    :align: center