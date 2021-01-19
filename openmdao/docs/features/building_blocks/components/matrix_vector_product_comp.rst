
.. _matrixvectorproductcomp_feature:

=======================
MatrixVectorProductComp
=======================

`MatrixVectorProductComp` performs a matrix-vector product.  It may be vectorized to provide the result at one or more points simultaneously.

.. math::

    \bar{b}_i = \left[ A_i \right] \bar{x}_i


MatrixVectorProductComp Options
-------------------------------

The default `vec_size` is 1, providing the matrix vector product of :math:`a` and :math:`x` at a single
point.

Other options for MatrixVectorProductComp allow the user to rename the input variables :math:`a` and :math:`x`
and the output :math:`b`, as well as specifying their units.

.. embed-options::
    openmdao.components.matrix_vector_product_comp
    MatrixVectorProductComp
    options


MatrixVectorProductComp Constructor
-----------------------------------

The call signature for the `MatrixVectorProductComp` constructor is:

.. automethod:: openmdao.components.matrix_vector_product_comp.MatrixVectorProductComp.__init__
    :noindex:


MatrixVectorProductComp Usage
-----------------------------

There are often situations when numerous products need to be computed, essentially in parallel.
You can reduce the number of components required by having one `MatrixVectorProductComp` perform multiple operations.
This is also convenient when the different operations have common inputs.

The ``add_product`` method is used to create additional products after instantiation.

.. automethod:: openmdao.components.matrix_vector_product_comp.MatrixVectorProductComp.add_product
   :noindex:


MatrixVectorProductComp Example
-------------------------------

The following code demonstrates the use of the MatrixVectorProductComp, finding the product
of a random 3x3 matrix `Mat` and a 3-vector `x` at 100 points simultaneously.

.. embed-code::
    openmdao.components.tests.test_matrix_vector_product_comp.TestFeature.test
    :layout: interleave


MatrixVectorProductComp Example with Multiple Products
------------------------------------------------------

When defining multiple products:

- An input name in one call to `add_product` may not be an output name in another call, and vice-versa.
- The units and shape of variables used across multiple products must be the same in each one.


.. embed-code::
    openmdao.components.tests.test_matrix_vector_product_comp.TestFeature.test_multiple
    :layout: interleave


.. tags:: MatrixVectorProductComp, Component
