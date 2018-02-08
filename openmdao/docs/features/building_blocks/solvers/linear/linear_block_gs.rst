.. _linearblockgs:

*************
LinearBlockGS
*************

LinearBlockGS uses Block Gauss-Seidel to solve the linear system. LinearBlockGS iterates until the linear
residual is below a tolerance, or the maximum number of iterations has been exceeded. As such,
it is generally usable for any system topology, and can handle cycles and implicit states
alike. It is not always the best solver to choose, however, and is known to diverge or plateau
on some problems. In such a case, you may need to use a solver such as
:ref:`ScipyKrylov <openmdao.solvers.linear.scipy_iter_solver.py>`.

LinearBlockGS is a block solver, so you can specify different linear solvers in the subsystems and they
will be utilized to solve the subsystem linear problem.

Note that systems without cycles or implicit states will converge in one iteration of Block Gauss-Seidel.

Here, we calculate the total derivatives across the Sellar system.

.. embed-test::
    openmdao.solvers.linear.tests.test_linear_block_gs.TestBGSSolverFeature.test_specify_solver


LinearBlockGS Options
---------------------

.. embed-options::
    openmdao.solvers.linear.linear_block_gs
    LinearBlockGS
    options

LinearBlockGS Option Examples
-----------------------------

**maxiter**

  This lets you specify the maximum number of Gauss-Seidel iterations to apply. In this example, we
  cut it back from the default, ten, down to two, so that it terminates a few iterations earlier and doesn't
  reach either of the specified absolute or relative tolerances.

  .. embed-test::
      openmdao.solvers.linear.tests.test_linear_block_gs.TestBGSSolverFeature.test_feature_maxiter

**atol**

  Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the linear residuals is calculated by calling `apply_linear`. If this norm value is lower than the absolute
  tolerance `atol`, the iteration will terminate.

  .. embed-test::
      openmdao.solvers.linear.tests.test_linear_block_gs.TestBGSSolverFeature.test_feature_atol

**rtol**

  Here, we set the relative tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the linear residuals is calculated by calling `apply_linear`. If the ratio of the currently calculated norm to the
  initial residual norm is lower than the relative tolerance `rtol`, the iteration will terminate.

  .. embed-test::
      openmdao.solvers.linear.tests.test_linear_block_gs.TestBGSSolverFeature.test_feature_rtol

.. tags:: Solver, LinearSolver
