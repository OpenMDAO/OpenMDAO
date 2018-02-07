.. _feature_specify_partials:

*****************************
Declaring Partial Derivatives
*****************************

If you know additional information about the structure of partial derivatives in your component
(for example, if an output does not depend on a particular input), you can use the :code:`declare_partials()`
method to inform the framework. This will allow the framework to be more efficient in terms of
memory and computation (especially if using a sparse :code:`AssembledJacobian`). This information
should be delcared in the `setup` method of your component.

.. automethod:: openmdao.core.component.Component.declare_partials
    :noindex:

Usage
-----

1. Specifying that a variable does not depend on another. Note that this is not typically required, because by default OpenMDAO assumes that all variables are independent.
However, in some cases it might be needed if a previous glob pattern matched a large set of variables and some sub-set of that needs to be marked as independent.

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompDependence.setup

2. Declaring multiple derivatives using glob patterns (see https://docs.python.org/3.6/library/fnmatch.html).

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompGlob.setup

3. Using the :code:`val` argument to set a constant partial derivative. Note that this is intended for cases when the derivative value is constant,
and hence the derivatives do not ever need to be recomputed in :code:`compute_partials`.
Here are several examples of how you can specify derivative values for differently-shaped partial derivative sub-Jacobians.

* Scalar [see :math:`\displaystyle\frac{\partial f}{\partial x}`]
* Dense Array [see :math:`\displaystyle\frac{\partial f}{\partial z}`]
* Nested List [see :math:`\displaystyle\frac{\partial g}{\partial y_1}` and
  :math:`\displaystyle\frac{\partial g}{\partial y_3}`]
* Sparse Matrix (see :ref:`Sparse Partial Derivatives doc <feature_sparse_partials>` for more details)
  [see :math:`\displaystyle\frac{\partial g}{\partial y_2}` and
  :math:`\displaystyle\frac{\partial g}{\partial x}`]

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.SimpleCompConst


.. tags:: PartialDerivatives
