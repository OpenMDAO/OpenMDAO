Specifying Partial Derivatives
==============================

If you know additional information about the structure of partial derivatives in your component,
say if an output does not depend on a particular input, you can use the :code:`declare_partials()`
method to inform the framework. This will allow the framework to be more efficient in terms of
memory and computation (especially if using a sparse :code:`AssembledJacobian`).

.. automethod:: openmdao.core.component.Component.declare_partials
    :noindex:

Usage
-----

1. Specifying that a variable does not depend on another.

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompDependence.initialize_partials

2. Specifying variables using glob patterns (see https://docs.python.org/3.6/library/fnmatch.html).

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompGlob.initialize_partials

3. Using the :code:`val` argument to set a constant partial derivative. Note that if the :code:`val` arugment is used,
then the partial derivative does not need to be calculated in :code:`compute_partial_derivs`.

* Scalar [see :math:`\displaystyle\frac{\partial f}{\partial x}`]
* Dense Array [see :math:`\displaystyle\frac{\partial f}{\partial z}`]
* Nested List [see :math:`\displaystyle\frac{\partial g}{\partial y_1}` and
  :math:`\displaystyle\frac{\partial g}{\partial y_3}`]
* Sparse Matrix (we will discuss these in more detail later)
  [see :math:`\displaystyle\frac{\partial g}{\partial y_2}` and
  :math:`\displaystyle\frac{\partial g}{\partial x}`]

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompConst

.. embed-test::
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_const_jacobian

.. tags:: Partial Derivatives

