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

.. embed-code::
    openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_basic
    :layout: interleave

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


Aitken relaxation
-------------------
This solver implements Aitken relaxation, as described in Algorithm 1 of this paper on aerostructual design optimization_.
The relaxation is turned off by default, but it may help convergence for more tightly coupled models.

.. _optimization: http://mdolab.engin.umich.edu/content/scalable-parallel-approach-aeroelastic-analysis-and-derivative

Residual Calculation
--------------------
The `Unified Derivatives Equations` are formulated so that explicit equations (via `ExplicitComponent`) are also expressed
as implicit relationships, and their residual is also calculated in "apply_linear", which runs the component a second time and
saves the difference in the output vector as the residual. However, this would require an extra call to `execute`, which is
inefficient for slower components. To elimimate the inefficiency of running the model twice every iteration the NonlinearBlockGS
driver saves a copy of the output vector and uses that to calculate the residual without rerunning the model. This does require
a little more memory, so if you are solving a model where memory is more of a concern than execution time, you can set the
"use_apply_nonlinear" option to True to use the original formulation that calls "apply_linear" on the subsystem.


NonlinearBlockGS Option Examples
--------------------------------

**maxiter**

  `maxiter` lets you specify the maximum number of Gauss-Seidel iterations to apply. In this example, we
  cut it back from the default, ten, down to two, so that it terminates a few iterations earlier and doesn't
  reach the specified absolute or relative tolerance.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_maxiter
      :layout: interleave

**atol**

  Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated one of two ways. If the "use_apply_linear" option
  is set to False (its default), then the norm is calculated by subtracting a cached previous value of the
  outputs from the current value.  If "use_apply_linear" is True, then the norm is calculated by calling
  apply_linear on all of the subsystems. In this case, `ExplicitComponents` are executed a second time.
  If this norm value is lower than the absolute tolerance `atol`, the iteration will terminate.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_atol
      :layout: interleave

**rtol**

  Here, we set the relative tolerance to a looser value that will trigger an earlier termination. After
  each iteration, the norm of the residuals is calculated one of two ways. If the "use_apply_linear" option
  is set to False (its default), then the norm is calculated by subtracting a cached previous value of the
  outputs from the current value.  If "use_apply_linear" is True, then the norm is calculated by calling
  apply_linear on all of the subsystems. In this case, `ExplicitComponents` are executed a second time.
  If the ratio of the currently calculated norm to the initial residual norm is lower than the relative tolerance
  `rtol`, the iteration will terminate.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_rtol
      :layout: interleave

.. tags:: Solver, NonlinearSolver
