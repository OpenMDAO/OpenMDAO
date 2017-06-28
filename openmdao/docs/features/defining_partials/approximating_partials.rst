Approximating Partial Derivatives
=================================

OpenMDAO allows you to specify analytic derivatives for your models, but it is not a requirement.
If certain partial derivatives are not available, you can ask the framework to approximate the
derivatives by using the :code:`approx_partials` method inside :code:`setup_partials`.

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

Approximating Semi-Total Derivatives
====================================

There are also times where it makes more sense to approximate the derivatives for an entire group in one shot. You can turn on
the approximation by calling `approx_total_derivs` on any `Group`.

.. automethod:: openmdao.core.group.Group.approx_total_derivs
    :noindex:

The default method is finite difference, and OpenMDAO
automatically figures out what derivatives are needed to populate the Jacobian. When total derivative approximation is turned
on in a group, all linear solves or derivative computations that are intiated above that Group's level are approximated for this group, and
for those calculations, the Group looks like a Component with an approximated Jacobian. However, the Jacobian contains total
derivatives rather than partial derivatives, so any implicit states that are contained within the group will not have their
partials exposed for convergence by solvers that are higher in the hierarchy. If you want to finite difference a Group that
contains implicit states somewhere inside, then you must have an appropriate solver (such as NewtonSolver) inside the group
to solve the implicit relationships.

Here is a classic example of where you might use an approximation like finite difference. In this example, we could just
approximate the partials on components CompOne and CompTwo separately. However, CompTwo has a vector input that is 25 wide,
so it would require 25 separate executions under finite difference. If we instead approximate the total derivatives on the
whole group, we only have one input, so just one extra execution.

.. embed-test::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_basic

The same arguments are used for both partial and total derivative approximation specifications. Here we set the finite difference
step size, the form to central differences, and the step_calc to relative instead of absolute.

.. embed-test::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_arguments