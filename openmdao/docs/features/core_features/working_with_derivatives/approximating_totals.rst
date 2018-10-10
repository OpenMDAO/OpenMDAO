.. _feature_declare_totals_approx:


Approximating Semi-Total Derivatives
====================================

There are times where it makes sense to approximate the derivatives for an entire group in one shot.
You can turn on the approximation by calling :code:`approx_totals` on any :code:`Group`.

.. automethod:: openmdao.core.group.Group.approx_totals
    :noindex:

The default method for approximating semi-total derivatives is the finite difference method. When
you call the :code:`approx_totals` method on a group, OpenMDAO will
generate an approximate Jacobian for the entire group during the linearization step before
derivatives are calculated. OpenMDAO automatically figures out
which inputs and output pairs are needed in this Jacobian. When :code:`solve_linear` is called from
any system that contains this system, the approximated Jacobian
is used for the derivatives in this system.

The derivatives approximated in this manner are total derivatives of outputs of the group with
respect to inputs. If any components in the group contain
implicit states, then you must have an appropriate solver (such as :code:`NewtonSolver`) inside the
group to solve the implicit relationships.

Here is a classic example of where you might use an approximation like finite difference. In this
example, we could just approximate the partials on components CompOne and CompTwo separately.
However, CompTwo has a vector input that is 25 wide, so it would require 25 separate executions
under finite difference. If we instead approximate the total derivatives on the
whole group, we only have one input, so just one extra execution.

.. embed-code::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_basic
    :layout: interleave

The same arguments are used for both partial and total derivative approximation specifications.
Here we set the finite difference `step` size, the `form` to central differences, and the
`step_calc` to relative instead of absolute.

.. embed-code::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_arguments
    :layout: interleave

Complex Step
------------

.. _cs_guidelines:

You can also complex step your model or group, though there are some important restrictions.

**All components must support complex calculations in solve_nonlinear:**
  Under complex step, a component’s inputs are complex, all stages of the calculation will operate
  on complex inputs to produce complex outputs, and the final value placed into outputs is complex.
  Most Python functions already support complex numbers, so pure Python components will generally
  satisfy this requirement. Take care with functions like :code:`abs`, which effectively squelches
  the complex part of the argument.

**If you complex step around a solver that requires gradients, the solver must not get its gradients from complex step:**
  When you complex step around a nonlinear solver that requires gradients (like Newton), the
  nonlinear solver must solve a complex linear system rather than a real one. Most of OpenMDAO's linear solvers
  (with the exception of `PetscKSP`) support the solution of such a system.  However, when you linearize the submodel
  to determine derivatives around a complex point, the application of complex step loses some of its robust properties
  when compared to real-valued finite difference (in particular, you get subtractive cancelation which causes
  increased inaccuracy for smaller stepsizes.) When OpenMDAO encounters this situation, it will warn the user, and the
  inner approximation will automatically switch over to using finite difference with default settings.

**Care must be taken with iterative solver tolerances; you may need to adjust the stepsize for complex step:**
  If you are using an interative nonlinear solver, and you don't converge it tightly, then the complex stepped
  linear system may have trouble converging as well. You may need to tighten the convergence of your solvers
  and increase the step size used for complex step. To prevent the nonlinear solvers from ignoring a tiny
  complex step, a tiny offset is added to the states to nudge it off the solution, allowing it to reconverge
  with the complex step. You can also turn this behavior off by setting the "cs_reconverge" to False.

  Similarly, an iterative linear solver may also require
  adjusting the stepsize, particularly if you are using the `ScipyKrylov` solver.

.. embed-code::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_basic_cs
    :layout: interleave
