Specifying Partial Derivatives
==============================

If you know additional information about the structure of partial derivatives in your component,
say if an output does not depend on a particular input, you can use the :code:`declare_partials()`
method to inform the framework. This will allow the framework to be more efficient in terms of
memory and computation (if using a sparse :code:`GlobalJacobian`).

.. automethod:: openmdao.core.component.Component.declare_partials
    :noindex:

Examples
--------

If two variables do not depend on each other, you can specify that they are not dependent.

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompDependence.initialize_variables

----

In addition to specifying specific variables, glob patterns may also be used
(see https://docs.python.org/3.6/library/fnmatch.html).

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompGlob.initialize_variables

----

If a particular partial derivative is constant, we can use the :code:`val` argument to specify what
that value. This derivative then does not need to be calculated in :code:`compute_jacobian`. This
value can take many forms:

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
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianFeatures.test_const_jacobian

----

Related Features
----------------
