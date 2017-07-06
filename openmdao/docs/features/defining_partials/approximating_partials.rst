Approximating Partial Derivatives
=================================

OpenMDAO allows you to specify analytic derivatives for your models, but it is not a requirement.
If certain partial derivatives are not available, you can ask the framework to approximate the
derivatives by using the :code:`approx_partials` method inside :code:`setup`.

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

The default method is for approximating semi-total derivatives is the finite difference method. When you call the `approx_total_derivs` method on a group, OpenMDAO will
generate an approximate Jacobian for the entire group during the linearization step before derivatives are calculated. OpenMDAO automatically figures out
which inputs and output pairs are needed in this Jacobian. When `solve_linear` is called from any system that contains this system, the approximated Jacobian
is used for the derivatives in this system.

The derivatives approximated in this matter are total derivatives of outputs of the group with respect to inputs. If any components in the group contain
implicit states, then you must have an appropriate solver (such as NewtonSolver) inside the group to solve the implicit relationships.

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