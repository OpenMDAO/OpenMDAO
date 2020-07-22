
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

VectorMagnitudeComp Constructor
-------------------------------

The call signature for the `VectorMagnitudeComp` constructor is:

.. automethod:: openmdao.components.vector_magnitude_comp.VectorMagnitudeComp.__init__
    :noindex:


VectorMagnitudeComp Usage
-------------------------

There are often situations when numerous magnitudes need to be computed, essentially in parallel.
You can reduce the number of components required by having one `VectorMagnitudeComp` perform multiple operations.
This is also convenient when the different operations have common inputs.

The ``add_magnitude`` method is used to create additional magnitude calculations after instantiation.

.. automethod:: openmdao.components.vector_magnitude_comp.VectorMagnitudeComp.add_magnitude
   :noindex:


VectorMagnitudeComp Example
---------------------------

In the following example VectorMagnitudeComp is used to compute the radius vector magnitude
given a radius 3-vector at 100 points simultaneously. Note the use of
`in_name` and `mag_name` to assign names to the inputs and outputs.
Units are assigned using `units`.  The units of the output magnitude are the same as those for
the input.

.. embed-code::
    openmdao.components.tests.test_vector_magnitude_comp.TestFeature.test
    :layout: interleave


VectorMagnitudeComp Example with Multiple Magnitudes
----------------------------------------------------

Note that, when defining multiple magnitudes, an input name in one call to `add_magnitude` may not be an
output name in another call, and vice-versa.


.. embed-code::
    openmdao.components.tests.test_vector_magnitude_comp.TestFeature.test_multiple
    :layout: interleave


.. tags:: VectorMagnitudeComp, Component
