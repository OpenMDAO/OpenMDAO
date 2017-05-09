Approximating Partial Derivatives
=================================

OpenMDAO allows you to specify analytic derivatives for your models, but it is not a requirement.
If certain partial derivatives are not available, you can ask the framework to approximate the
derivatives by using the :code:`approx_partials` method inside :code:`initialize_partials`.

.. automethod:: openmdao.core.component.Component.approx_partials
    :noindex:

Usage
-----

1. Much like in :code:`declare_partials`, you may use glob patterns as arguments to :code:`to` and :code:`wrt`.

.. embed-test::
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_fd_glob

2. For finite difference approximations (:code:`method='fd'`), we have two (optional) parameters: the form and step size. The form should be one of the following:
        - :code:`form='forward'` (default): Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x+\delta, y) - f(x,y)}{||\delta||}`. Error scales like :math:`||\delta||`.
        - :code:`form='backward'`: Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x,y) - f(x-\delta, y) }{||\delta||}`. Error scales like :math:`||\delta||`.
        - :code:`form='central'`: Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x+\delta, y) - f(x-\delta,y)}{2||\delta||}`. Error scales like :math:`||\delta||^2`, but requires an extra function evaluation.

The step size can be any non-zero number, but should be positive (one can change the form to perform backwards finite difference formulas), small enough to reduce truncation error, but large enough to avoid round-off error. Choosing a step size can be highly problem dependent, but for double precision floating point numbers and reasonably bounded derivatives, :math:`10^{-6}` can be a good place to start.

.. embed-test::
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_fd_options