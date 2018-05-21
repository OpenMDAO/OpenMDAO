.. index:: Linear System Example

****************
LinearSystemComp
****************

The LinearSystemComp solves the linear system Ax = b where A and b are inputs, and x is the output.

LinearSystemComp Options
------------------------

.. embed-options::
    openmdao.components.linear_system_comp
    LinearSystemComp
    options

LinearSystemComp Example
------------------------

.. embed-code::
    openmdao.components.tests.test_linear_system_comp.TestLinearSystemComp.test_feature_basic
    :layout: interleave

This component can also be vectorized to either solve a single linear system with multiple right hand sides, or to solve
multiple independent linear systems.

You can solve multiple right hand sides by setting the "vec_size" argument, giving it the number of right hand sides. When
you do this, the LinearSystemComp creates an input for "b" such that each row of b is solved independently.

.. embed-code::
    openmdao.components.tests.test_linear_system_comp.TestLinearSystemComp.test_feature_vectorized
    :layout: interleave

To solve multiple linear systems, you just need to set the "vectorize_A" option or argument to True. The A
matrix is now a 3-dimensional matrix where the first dimension is the number of linear systems to solve.

.. embed-code::
    openmdao.components.tests.test_linear_system_comp.TestLinearSystemComp.test_feature_vectorized_A
    :layout: interleave

.. tags:: LinearSystemComp, Component
