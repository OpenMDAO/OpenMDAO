:orphan:

.. _nlnewton:

Nonlinear Solver: NewtonSolver
==============================

The `NewtonSolver` solver implements Newton's method to solve the system that contains it. This
is the most general solver in OpenMDAO in that it can solve any topology including cyclic
connections and implicit states in the system or subsystems. Newton's method requires derivatives,
so a linear solver can also be specified. By default, the NewtonSolver uses the linear solver
that is slotted in the containing system.

.. embed-test::
    openmdao.solvers.tests.test_nl_newton.TestNewtonFeatures.test_feature_basic

Most of the solver in OpenMDAO operate hierarchically in that you can use solvers on subgroups
to subdivide the calculation effort. However, the NewtonSolver is an exception. It does not
call `solve_nonlinear` on its subsystems, nor does it pass data along the connections. Instead,
the Newton solver sets all inputs in all systems and subsystems that it contains, as it follows
the gradient driving the residuals to convergence.  After each iteration, the iteration count and the residual norm are
checked to see if termination has been satisfied.

Settings: maxiter
-----------------

This lets you specify the maximum number of Newton iterations to apply. In this example, we
cut it back from the default (10) to 2 so that it terminates a few iterations earlier and doesn't
reach the specified absolute or relative tolerance.

.. embed-test::
    openmdao.solvers.tests.test_nl_newton.TestNewtonFeatures.test_feature_maxiter

Settings: atol
--------------

Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
components and `evaluate` on explicit components. If this norm value is lower than the absolute
tolerance `atol`, the iteration will terminate.

.. embed-test::
    openmdao.solvers.tests.test_nl_newton.TestNewtonFeatures.test_feature_atol

Settings: rtol
--------------

Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
components and `evaluate` on explicit components. If the ratio of the currently calculated norm to the
initial residual norm is lower than the relative tolerance `rtol`, the iteration will terminate.

.. embed-test::
    openmdao.solvers.tests.test_nl_newton.TestNewtonFeatures.test_feature_rtol

Specifying a Linear Solver
--------------------------

We can choose a different linear solver for calculating the Newton step. The default is to use the
linear solver that was specified on the containing system, which by default is LinearBlockGS. Here,
we modify the model to use :ref:`DirectSolver <usr_openmdao.solvers.ln_direct.py>` instead.

.. embed-test::
    openmdao.solvers.tests.test_nl_newton.TestNewtonFeatures.test_feature_ln_solver

Specifying a Linesearch algorithm
---------------------------------