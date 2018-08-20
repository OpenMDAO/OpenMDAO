.. _nlbjac:

*****************
NonlinearBlockJac
*****************

NonlinearBlockJac is a nonlinear solver that uses the block Jacobi method to solve
the system. When to choose this solver over :ref:`NonlinearBlockGS <openmdao.solvers.nonlinear.nonlinear_block_gs.py>`
is an advanced topic, but it is valid for systems that satisfy the same conditions:

1. System (or subsystem) contains a cycle, though subsystems may.
2. System does not contain any implicit states, though subsystems may.

Note that you may not know if you satisfy the second condition, so choosing a solver can be "trial and error." If
NonlinearBlockJac doesn't work, then you will need to use :ref:`NewtonSolver <openmdao.solvers.nonlinear.newton.py>`.

The main difference over `NonlinearBlockGS` is that data passing is delayed until after all subsystems have been
executed.

Here, we choose NonlinearBlockJac to solve the Sellar problem, which has two components with a
cyclic dependency, has no implicit states, and works very well with Jacobi.

.. embed-code::
    openmdao.solvers.nonlinear.tests.test_nonlinear_block_jac.TestNLBlockJacobi.test_feature_basic
    :layout: interleave

This solver runs all of the subsystems each iteration, but just passes the data along all connections
simultaneously once per iteration. After each iteration, the iteration count and the residual norm are
checked to see if termination has been satisfied.

You can control the termination criteria for the solver using the following options:

NonlinearBlockJac Options
-------------------------

.. embed-options::
    openmdao.solvers.nonlinear.nonlinear_block_jac
    NonlinearBlockJac
    options

NonlinearBlockJac Option Examples
---------------------------------

**maxiter**

  This lets you specify the maximum number of Jacobi iterations to apply. In this example, we
  cut it back from the default, ten, down to two, so that it terminates a few iterations earlier and doesn't
  reach the specified absolute or relative tolerance.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_jac.TestNLBlockJacobi.test_feature_maxiter
      :layout: interleave

**atol**

  Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
  components and `evaluate` on explicit components. If this norm value is lower than the absolute
  tolerance `atol`, the iteration will terminate.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_jac.TestNLBlockJacobi.test_feature_atol
      :layout: interleave

**rtol**

  Here, we set the relative tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated by calling `apply_nonlinear` on implicit
  components and `evaluate` on explicit components. If the ratio of the currently calculated norm to the
  initial residual norm is lower than the relative tolerance `rtol`, the iteration will terminate.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_jac.TestNLBlockJacobi.test_feature_rtol
      :layout: interleave

.. tags:: Solver, NonlinearSolver
