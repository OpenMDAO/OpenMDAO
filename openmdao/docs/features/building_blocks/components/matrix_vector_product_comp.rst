
.. _matrixvectorproductcomp_feature:

=======================
MatrixVectorProductComp
=======================

`MatrixVectorProductComp` performs a matrix-vector product.  It may be vectorized to provide the result at one or more points simultaneously.

.. math::

    \bar{c}_i = \left[ A_i \right] \bar{b}_i

The default `vec_size` is 1, providing the matrix vector product of :math:`a` and :math:`b` at a single
point.

Options for MatrixVectorProductComp allow the user to rename the input variables :math:`a` and :math:`b`
and the output :math:`c`, as well as specifying their units.


.. embed-options::
    openmdao.components.matrix_vector_product_comp
    MatrixVectorProductComp
    metadata

The following code demonstrates the use of the MatrixVectorProductComp, finding the product
of a random 3x3 matrix `M` and a 3-vector `x` at 100 points simultaneously.

.. embed-code::
    openmdao.components.tests.test_matrix_vector_product_comp.TestForDocs.test
