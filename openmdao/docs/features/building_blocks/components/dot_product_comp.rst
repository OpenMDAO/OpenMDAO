
.. _dotproductcomp_feature:

**************
DotProductComp
**************

`DotProductComp` performs a dot-product between two compatible inputs.  It may be vectorized to provide the result at one or more points simultaneously.

.. math::

    c_i = \bar{a}_i \cdot \bar{b}_i

The default `vec_size` is 1, providing the dot product of :math:`a` and :math:`b` at a single
point.  The lengths of :math:`a` and :math:`b` are provided by option `length`.

Options for DotProductComp allow the user to rename the input variables :math:`a` and :math:`b`
and the output :math:`c`, as well as specifying their units.


.. embed-options::
    openmdao.components.dot_product_comp
    DotProductComp
    metadata

In the following example DotProductComp is used to compute instantaneous power as the
dot product of force and velocity at 100 points simultaneously.  Note the use of
`a_name`, `b_name`, and `c_name` to assign names to the inputs and outputs.
Units are assigned using `a_units`, `b_units`, and `c_units`.
Note that no internal checks are performed to ensure that `c_units` are consistent
with `a_units` and `b_units`.

.. embed-code::
    openmdao.components.tests.test_cross_product_comp.TestForDocs.test

.. tags:: DotProductComp, Component
