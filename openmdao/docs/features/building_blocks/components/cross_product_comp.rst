
.. _crossproductcomp_feature:

.. meta::
   :description: OpenMDAO Feature doc for CrossProductComp, which performs a cross product on two inputs
   :keywords: cross product, CrossProductComp

================
CrossProductComp
================

`CrossProductComp` performs a cross product between two 3-vector inputs.  It may be vectorized to provide the result at one or more points simultaneously.

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


CrossProductComp Constructor
----------------------------

The call signature for the `CrossProductComp` constructor is:

.. automethod:: openmdao.components.cross_product_comp.CrossProductComp.__init__
    :noindex:


CrossProductComp Usage
----------------------

There are often situations when numerous products need to be computed, essentially in parallel.
You can reduce the number of components required by having one `CrossProductComp` perform multiple operations.
This is also convenient when the different operations have common inputs.

The ``add_product`` method is used to create additional products after instantiation.

.. automethod:: openmdao.components.cross_product_comp.CrossProductComp.add_product
   :noindex:


CrossProductComp Example
------------------------

In the following example CrossProductComp is used to compute torque as the
cross product of force (:math:`F`) and radius (:math:`r`) at 100 points simultaneously.
Note the use of `a_name`, `b_name`, and `c_name` to assign names to the inputs and outputs.
Units are assigned using `a_units`, `b_units`, and `c_units`.
Note that no internal checks are performed to ensure that `c_units` are consistent
with `a_units` and `b_units`.

.. embed-code::
    openmdao.components.tests.test_cross_product_comp.TestFeature.test
    :layout: interleave


DotProductComp Example with Multiple Products
---------------------------------------------

When defining multiple products:

- An input name in one call to `add_product` may not be an output name in another call, and vice-versa.
- The units and shape of variables used across multiple products must be the same in each one.


.. embed-code::
    openmdao.components.tests.test_cross_product_comp.TestFeature.test_multiple
    :layout: interleave


.. tags:: CrossProductComp, Component
