Specifying Partial Derivatives
==============================

If you know additional information about the structure of partial derivatives in your component,
say if an output does not depend on a particular input, you can use the :code:`declare_partials()`
method to inform the framework. This will allow the framework to be more efficient

.. automethod:: openmdao.core.component.Component.declare_partials
    :noindex:

Examples
--------

If two variables do not depend on each other, you can specify that they are not dependent.

.. embed-test::
    openmdao.jacobians.test_jacobian_features.TestJacobianFeatures.