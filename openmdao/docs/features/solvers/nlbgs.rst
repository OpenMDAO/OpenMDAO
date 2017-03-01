:orphan:

.. _nlbgs:

Nonlinear Solver: NonlinearBlockGS
==================================

The NonlinearBlockGS solver applies Gauss Seidel (also known as Fixed Point Iteration) to the
components and subsystems in the system. This is mainly used to solve cyclic connections. You
should try this solver for systems that satisfy the following conditions:

1. System (or subsystem) contains a cycle.
2. Function over the cycle satisfies Lipschitz condition with L<1.
3. System does not contain any implicit states.

Note that you may not know if you satisfy the second condition, so chosing a solver can be "trial and error." If
NonlinearBlockGS doesn't work, then you will need to use :ref:`NewtonSolver <usr_openmdao.solvers.nl_newton.py>`.

Here, we choose the NonlinearBlockGS to solve the Sellar problem, which has two components with a
cyclic dependency, has no implicit states, and works very well with Gauss Seidel.

.. embed-test::
    openmdao.solvers.tests.test_nl_bgs.TestNLBGaussSeidel.test_feature_basic

This solver runs all of the subsystems each iteration, passing data along all connections
including the cyclic ones. After each iteration, the iteration count and the residual norm are
checked to see if termination has been satisfied.

You can control the termination criteria for the solver using the following options:

Settings: maxiter
-----------------

This lets you specify the maximun number of Gauss Seidel iterations to apply. In this example, we
cut it back from the default (10) to 2 so that it terminates a few iterations earlier and doesn't
reach the specified absolute or relative tolerance..

.. embed-test::
    openmdao.solvers.tests.test_nl_bgs.TestNLBGaussSeidel.test_feature_maxiter

Settings: atol
--------------

Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
components and `evaluate` on explicit components. If this norm value is lower than the absolute
tolerance `atol`, the iteration will terminate.

.. embed-test::
    openmdao.solvers.tests.test_nl_bgs.TestNLBGaussSeidel.test_feature_atol

Settings: rtol
--------------

Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
components and `evaluate` on explicit components. If the ratio of the currently calculated norm to the
initial residual norm is lower than the relative tolerance `rtol`, the iteration will terminate.

.. embed-test::
    openmdao.solvers.tests.test_nl_bgs.TestNLBGaussSeidel.test_feature_rtol
