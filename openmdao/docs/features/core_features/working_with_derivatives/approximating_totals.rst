.. _feature_declare_totals_approx:


Approximating Semi-Total Derivatives
====================================

There are times where it makes sense to approximate the derivatives for an entire group in one shot. You can turn on
the approximation by calling :code:`approx_totals` on any :code:`Group`.

.. automethod:: openmdao.core.group.Group.approx_totals
    :noindex:

The default method for approximating semi-total derivatives is the finite difference method. When you call the :code:`approx_totals` method on a group, OpenMDAO will
generate an approximate Jacobian for the entire group during the linearization step before derivatives are calculated. OpenMDAO automatically figures out
which inputs and output pairs are needed in this Jacobian. When :code:`solve_linear` is called from any system that contains this system, the approximated Jacobian
is used for the derivatives in this system.

The derivatives approximated in this matter are total derivatives of outputs of the group with respect to inputs. If any components in the group contain
implicit states, then you must have an appropriate solver (such as :code:`NewtonSolver`) inside the group to solve the implicit relationships.

Here is a classic example of where you might use an approximation like finite difference. In this example, we could just
approximate the partials on components CompOne and CompTwo separately. However, CompTwo has a vector input that is 25 wide,
so it would require 25 separate executions under finite difference. If we instead approximate the total derivatives on the
whole group, we only have one input, so just one extra execution.

.. embed-test::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_basic

The same arguments are used for both partial and total derivative approximation specifications. Here we set the finite difference
`step` size, the `form` to central differences, and the `step_calc` to relative instead of absolute.

.. embed-test::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_arguments

Complex Step
------------

You can also complex step your model or group, though there are some important restrictions.

**All components must support complex calculations in solve_nonlinear:**
  Under complex step, a component’s inputs are complex, all stages of the calculation will operate on complex inputs to produce
  complex outputs, and the final value placed into outputs is complex. Most Python functions already support complex numbers, so pure
  Python components will generally satisfy this requirement. Take care with functions like :code:`abs`, which effectively squelches the complex
  part of the argument.

**Solvers like Newton that require gradients are not supported:**
  Complex stepping a model causes it to run with complex inputs. When there is a nonlinear solver at some level, the solver must be
  able to converge. Some solvers such as :code:`NonlinearBlockGS` can handle this. However, the Newton solver must linearize and initiate a
  gradient solve about a complex point. This is not possible to do at present (though we are working on some ideas to make this work.)

.. embed-test::
    openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_basic_cs