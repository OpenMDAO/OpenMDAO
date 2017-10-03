.. _feature_delare_partials_approx:

*********************************
Approximating Partial Derivatives
*********************************

OpenMDAO allows you to specify analytic derivatives for your models, but it is not a requirement.
If certain partial derivatives are not available, you can ask the framework to approximate the
derivatives by using the :code:`declare_partials` method inside :code:`setup` and giving it a
method that is either 'fd' for finite diffference or 'cs' for complex step.

.. automethod:: openmdao.core.component.Component.declare_partials
    :noindex:

Usage
-----

1. You may use glob patterns as arguments to :code:`to` and :code:`wrt`.

.. embed-test::
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_fd_glob

2. For finite difference approximations (:code:`method='fd'`), we have three (optional) parameters: the form, step size, and the step_calc The form should be one of the following:
        - :code:`form='forward'` (default): Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x+\delta, y) - f(x,y)}{||\delta||}`. Error scales like :math:`||\delta||`.
        - :code:`form='backward'`: Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x,y) - f(x-\delta, y) }{||\delta||}`. Error scales like :math:`||\delta||`.
        - :code:`form='central'`: Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x+\delta, y) - f(x-\delta,y)}{2||\delta||}`. Error scales like :math:`||\delta||^2`, but requires an extra function evaluation.

The step size can be any non-zero number, but should be positive (one can change the form to perform backwards finite difference formulas), small enough to reduce truncation error, but large enough to avoid round-off error. Choosing a step size can be highly problem dependent, but for double precision floating point numbers and reasonably bounded derivatives, :math:`10^{-6}` can be a good place to start.
The step_calc can be either 'abs' for absoluate or 'rel' for relative. It determines whether the stepsize ie absolute or a percentage of the input value.

.. embed-test::
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_fd_options

Complex Step
------------

If you have a pure python component (or an external code that can support complex inputs and outputs) then you can also choose to use
complex step to calculate the Jacobian of that component. This will give more accurate derivatives that are insensitive to the step size.
Like finite difference, complex step runs your component using the apply_nonlinear or solve_nonlinear functions, but it applies a step
in the complex direction. You can activate it using the :code:`declare_partials` method inside :code:`setup` and giving it a method of 'cs'.
In many cases, this will require no other changes to your code, as long as all of the calculation in your solve_nonlinear and
apply_nonlinear support complex numbers. During a complex step, the incoming inputs vector will return a complex number when a variable
is being stepped. Likewise, the outputs and residuals vectors will accept complex values. If you are allocating temporary numpy arrays,
remember to conditionally set their dtype based on the dtype in the outputs vector.

Here is how to turn on complex step for all input/output pairs in the Sellar problem:

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarDis1CS

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarDis2CS


Approximating Semi-Total Derivatives
====================================

There are also times where it makes more sense to approximate the derivatives for an entire group in one shot. You can turn on
the approximation by calling `approx_totals` on any `Group`.

.. automethod:: openmdao.core.group.Group.approx_totals
    :noindex:

The default method is for approximating semi-total derivatives is the finite difference method. When you call the `approx_totals` method on a group, OpenMDAO will
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

Complex Step
------------

You can also complex step your model or group, though there are some important restrictions.

**All components must support complex calculations in solve_nonlinear:**
  Under complex step, a component’s inputs are complex, all stages of the calculation will operate on complex inputs to produce
  complex outputs, and the final value placed into outputs is complex. Most Python functions already support complex numbers, so pure
  Python components will generally satisfy this requirement. Take care with functions like abs, which effectively squelches the complex
  part of the argument.

**Solvers like Newton that require gradients are not supported:**
  Complex stepping a model causes it to run with complex inputs. When there is a nonlinear solver at some level, the solver must be
  able to converge. Some solvers such as NonlinearBlockGS can handle this. However, the Newton solver must linearize and initiate a
  gradient solve about a complex point. This is not possible to do at present (though we are working on some ideas to make this work.)

.. embed-test::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_basic_cs