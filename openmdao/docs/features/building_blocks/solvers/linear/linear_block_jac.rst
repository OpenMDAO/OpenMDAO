.. _linearblockjac:

**************
LinearBlockJac
**************

LinearBlockJac uses the block Jacobi method to solve the linear system. The method is similar to that used by the
:ref:`LinearBlockGS <openmdao.solvers.linear.linear_block_gs.py>` solver, except that it propagates the derivatives from outputs
to inputs only once per iteration. When to choose this solver over the other ones is an advanced topic.

LinearBlockJac is a block solver, so you can specify different linear solvers in the subsystems and they
will be utilized to solve the subsystem linear problem.

Here, we calculate the total derivatives across the Sellar system.

.. embed-code::
    openmdao.solvers.linear.tests.test_linear_block_jac.TestBJacSolverFeature.test_specify_solver
    :layout: interleave

LinearBlockJac Options
----------------------

.. embed-options::
    openmdao.solvers.linear.linear_block_jac
    LinearBlockJac
    options

LinearBlockJac Option Examples
------------------------------

**maxiter**

  This lets you specify the maximum number of Gauss-Seidel iterations to apply. In this example, we
  cut it back from the default, ten, down to five, so that it terminates a few iterations earlier and doesn't  reach the specified absolute or relative tolerance. Note that due to the delayed transfer of
  information, this takes more iterations to converge than the LinearBlockGS solver.

  .. embed-code::
      openmdao.solvers.linear.tests.test_linear_block_jac.TestBJacSolverFeature.test_feature_maxiter
      :layout: interleave

**atol**

  Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the linear residuals is calculated by calling `apply_linear`. If this norm value is lower than the absolute
  tolerance `atol`, the iteration will terminate.

  .. embed-code::
      openmdao.solvers.linear.tests.test_linear_block_jac.TestBJacSolverFeature.test_feature_atol
      :layout: interleave

**rtol**

  Here, we set the relative tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the linear residuals is calculated by calling `apply_linear`. If the ratio of the currently calculated norm to the
  initial residual norm is lower than the relative tolerance `rtol`, the iteration will terminate.

  .. embed-code::
      openmdao.solvers.linear.tests.test_linear_block_jac.TestBJacSolverFeature.test_feature_rtol
      :layout: interleave


.. tags:: Solver, LinearSolver
