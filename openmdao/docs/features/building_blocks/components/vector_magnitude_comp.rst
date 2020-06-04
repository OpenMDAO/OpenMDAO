
.. _vectormagnitudecomp_feature:

*******************
VectorMagnitudeComp
*******************

`VectorMagnitudeComp` computes the magnitude (L2 norm) of a single input of some given length.
It may be vectorized to provide the result at one or more points simultaneously.

.. math::

    \lvert a_i \rvert = \sqrt{\bar{a}_i \cdot \bar{a}_i}

VectorMagnitudeComp Options
---------------------------

The default `vec_size` is 1, providing the magnitude of :math:`a` at a single
point.  The length of :math:`a` is provided by option `length`.

Other options for VectorMagnitudeComp allow the user to rename the input variable :math:`a`
and the output :math:`a_mag`, as well as specifying their units.


.. embed-options::
    openmdao.components.vector_magnitude_comp
    VectorMagnitudeComp
    options

VectorMagnitudeComp Example
---------------------------

In the following example VectorMagnitudeComp is used to compute the radius vector magnitude
given a radius 3-vector at 100 points simultaneously. Note the use of
`in_name` and `mag_name` to assign names to the inputs and outputs.
Units are assigned using `units`.  The units of the output magnitude are the same as those for
the input.

.. embed-code::
    openmdao.components.tests.test_vector_magnitude_comp.TestFeature.test

.. tags:: VectorMagnitudeComp, Component
