.. _nlnewton:

************
NewtonSolver
************

NewtonSolver implements Newton's method to solve the system that contains it. This
is the most general solver in OpenMDAO, in that it can solve any topology including cyclic
connections and implicit states in the system or subsystems. Newton's method requires derivatives,
so a linear solver can also be specified. By default, NewtonSolver uses the linear solver
that is slotted in the containing system.

.. embed-test::
    openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_basic

Most of the solvers in OpenMDAO operate hierarchically, in that you can use solvers on subgroups
to subdivide the calculation effort. However, NewtonSolver is an exception. It does not
call `solve_nonlinear` on its subsystems, nor does it pass data along the connections. Instead,
the Newton solver sets all inputs in all systems and subsystems that it contains, as it follows
the gradient, driving the residuals to convergence.  After each iteration, the iteration count and the residual norm are
checked to see if termination has been satisfied.

NewtonSolver Options
--------------------

.. embed-options::
    openmdao.solvers.nonlinear.newton
    NewtonSolver
    options

NewtonSolver Option Examples
----------------------------

**maxiter**

  :code:`maxiter` lets you specify the maximum number of Newton iterations to apply. In this example, we
  cut it back from the default, ten, down to two, so that it terminates a few iterations earlier and doesn't
  reach the specified absolute or relative tolerance.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_maxiter

**atol**

  Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
  components and `evaluate` on explicit components. If this norm value is lower than the absolute
  tolerance `atol`, the iteration will terminate.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_atol

**rtol**

  Here, we set the relative tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
  components and `evaluate` on explicit components. If the ratio of the currently calculated norm to the
  initial residual norm is lower than the relative tolerance `rtol`, the iteration will terminate.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_rtol

**solve_subsystems**

  If you set this option to True, NewtonSolver will call `solve_nonlinear` on all of its subsystems. You can
  use this to solve difficult multi-level problems by attaching solvers to subsystems. This assures that those
  subsystems will already be in an internally solved state when the Newton solver goes to solve it.

  This example shows two instances of the Sellar model, which we have connected together to form a larger cycle.
  We specify a Newton solver in each Sellar subgroup as well as a top-level Newton solver, which we tell to solve
  its subsystems.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_solve_subsystems_basic

**max_sub_solves**

  This option is used in conjunction with the "solve_subsystems" option. It controls the number of iterations for which
  NewtonSolver will allow subsystems to solve themselves. When the iteration count exceeds `max_sub_solves`,  Newton
  returns to its default behavior.

  For example, if you set `max_sub_solves` to zero, then the solvers on subsystems are executed during the initial
  evaluation, but not during any subsequent iteration.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_max_sub_solves

**err_on_maxiter**

  If you set this to True, then when the solver hits the iteration limit without meeting the tolerance criteria, it
  will raise an AnalysisError exception. This is mainly important when coupled with a higher-level solver or
  driver (e.g., `pyOptSparseDriver`)that can handle the AnalysisError by adapting the stepsize and retrying.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_err_on_maxiter

  This feature can be set on any iterative nonlinear or linear solver.

Specifying a Linear Solver
--------------------------

We can choose a different linear solver for calculating the Newton step by setting the `linear_solver` attribute. The default is to use the
linear solver that was specified on the containing system, which by default is LinearBlockGS. In the following example,
we modify the model to use :ref:`DirectSolver <openmdao.solvers.linear.direct.py>` instead.

.. embed-test::
    openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_linear_solver

Specifying a Line Search Algorithm
----------------------------------

NewtonSolver has a `linesearch` attribute, which supports specification of a supplemental algorithm that can find a better point
along the Newton search direction. This is typically used for cases where we have declared upper
or lower bounds on some of the model outputs and we want to prevent Newton from moving into this
non-feasible space during iteration. An algorithm that does this is called a line search.

By default, NewtonSolver does not perform a line search. We will show how to specify one. First,
let's set up a problem that has implicit bounds on one of its states.

.. embed-code::
    openmdao.test_suite.components.implicit_newton_linesearch.ImplCompTwoStates

In this component, the state "z" is only valid between 1.5 and 2.5, while the other state is valid
everywhere. You can verify that if NewtonSolver is used with no backtracking specified, the solution
violates the bounds on "z".  Here, we specify :ref:`ArmijoGoldsteinLS <openmdao.solvers.linesearch.backtracking.py>`
as our line search algorithm, and we get a solution on the lower bounds for "z".

.. embed-test::
    openmdao.solvers.linesearch.tests.test_backtracking.TestFeatureLineSearch.test_feature_specification

.. tags:: Solver, NonlinearSolver
