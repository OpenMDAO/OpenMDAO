
.. _crossproductcomp_feature:

================
CrossProductComp
================

`CrossProductComp` performs a cross-product between two 3-vector inputs.  It may be vectorized to provide the result at one or more points simultaneously.

.. math::

    c_i = \bar{a}_i \times \bar{b}_i

The first dimension of the inputs holds the vectorized dimension.
The default `vec_size` is 1, providing the cross product of :math:`a` and :math:`b` at a single
point.  The lengths of :math:`a` and :math:`b` at each point must be 3.

The shape of :math:`a` and :math:`b` will always be `(vec_size, 3)`, but the connection rules
of OpenMDAO allow the incoming connection to have shape `(3,)` when `vec_size` is 1, since
the storage order of the underlying data is the same.  The output vector `c` of
CrossProductComp will always have shape `(vec_size, 3)`.

CrossProductComp Options
------------------------

Options for CrossProductComp allow the user to rename the input variables :math:`a` and :math:`b`
and the output :math:`c`, as well as specifying their units.

.. embed-options::
    openmdao.components.cross_product_comp
    CrossProductComp
    options

CrossProductComp Example
------------------------

In the following example DotProductComp is used to compute torque as the
cross product of force (:math:`F`) and radius (:math:`r`) at 100 points simultaneously.
Note the use of `a_name`, `b_name`, and `c_name` to assign names to the inputs and outputs.
Units are assigned using `a_units`, `b_units`, and `c_units`.
Note that no internal checks are performed to ensure that `c_units` are consistent
with `a_units` and `b_units`.

.. embed-code::
    openmdao.components.tests.test_cross_product_comp.TestForDocs.test

.. tags:: CrossProductComp, Component
