.. _nlbgs:

****************
NonlinearBlockGS
****************

NonlinearBlockGS applies Block Gauss-Seidel (also known as fixed-point iteration) to the
components and subsystems in the system. This is mainly used to solve cyclic connections. You
should try this solver for systems that satisfy the following conditions:

1. System (or subsystem) contains a cycle, though subsystems may.
2. System does not contain any implicit states, though subsystems may.

NonlinearBlockGS is a block solver, so you can specify different nonlinear solvers in the subsystems and they
will be utilized to solve the subsystem nonlinear problem.

Note that you may not know if you satisfy the second condition, so choosing a solver can be a trial-and-error proposition. If
NonlinearBlockGS doesn't work, then you will need to use :ref:`NewtonSolver <openmdao.solvers.nonlinear.newton.py>`.

Here, we choose NonlinearBlockGS to solve the Sellar problem, which has two components with a
cyclic dependency, has no implicit states, and works very well with Gauss-Seidel.

.. embed-test::
    openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_basic

This solver runs all of the subsystems each iteration, passing data along all connections
including the cyclic ones. After each iteration, the iteration count and the residual norm are
checked to see if termination has been satisfied.

You can control the termination criteria for the solver using the following options:

NonlinearBlockGS Options
------------------------

.. embed-options::
    openmdao.solvers.nonlinear.nonlinear_block_gs
    NonlinearBlockGS
    options

NonlinearBlockGS Option Examples
--------------------------------

**maxiter**

  `maxiter` lets you specify the maximum number of Gauss-Seidel iterations to apply. In this example, we
  cut it back from the default, ten, down to two, so that it terminates a few iterations earlier and doesn't
  reach the specified absolute or relative tolerance.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_maxiter

**atol**

  Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
  components and `evaluate` on explicit components. If this norm value is lower than the absolute
  tolerance `atol`, the iteration will terminate.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_atol

**rtol**

  Here, we set the relative tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
  components and `evaluate` on explicit components. If the ratio of the currently calculated norm to the
  initial residual norm is lower than the relative tolerance `rtol`, the iteration will terminate.

  .. embed-test::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_rtol

.. tags:: Solver, NonlinearSolver
