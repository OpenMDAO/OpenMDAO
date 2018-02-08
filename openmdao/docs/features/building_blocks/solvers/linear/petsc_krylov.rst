.. _petscKrylov:

***********
PETScKrylov
***********

PETScKrylov is an iterative linear solver that wraps the linear solution methods found in PETSc via petsc4py.
The default method is "fgmres", or the Flexible Generalized Minimal RESidual method, though you may choose any of
the other methods in PETSc. This linear solver is capable of handling any system topology
effectively. It also solves all subsystems below it in the hierarchy, so assigning different solvers to
subsystems will have no effect on the solution at this level.

This solver works under MPI, so it is a good alternative to
:ref:`ScipyKrylov <openmdao.solvers.linear.scipy_iter_solver.py>`.
This solver is also re-entrant, so there are no problems if it is nested during preconditioning.

Here, we calculate the total derivatives across the Sellar system.

.. embed-test::
    openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_specify_solver

PETScKrylov Options
-------------------

.. embed-options::
    openmdao.solvers.linear.petsc_ksp
    PETScKrylov
    options

PETScKrylov Option Examples
---------------------------

**maxiter**

  `maxiter` lets you specify the maximum number of GMRES (or other algorithm) iterations to apply. The default maximum is 100, which
  is much higher than the other linear solvers because each multiplication by the system Jacobian is considered
  to be an iteration. You may have to decrease this value if you have a coupled system that is converging
  very slowly. (Of course, in such a case, it may be better to add a preconditioner.)  Alternatively, you
  may have to raise `maxiter` if you have an extremely large number of components in your system (a 1000-component
  ring would need 1000 iterations just to make it around once.)

  This example shows what happens if you set `maxiter` too low (the derivatives should be nonzero, but it stops too
  soon.)

  .. embed-test::
      openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_feature_maxiter

**atol**

  The absolute convergence tolerance, the absolute size of the (possibly preconditioned) residual norm.

  You may need to adjust this setting if you have abnormally large or small values in your global Jacobian.

  .. embed-test::
      openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_feature_atol

**rtol**

  The relative convergence tolerance, the relative decrease in the (possibly preconditioned) residual norm.

  .. embed-test::
      openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_feature_rtol

**ksp_type**

  You can specify which PETSc algorithm to use in place of 'fgmres' by settng the "ksp_type" in the options
  dictionary.  Here, we use 'gmres' instead.

  .. embed-test::
      openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_specify_ksp_type

.. _petsckrylov_precon:

Specifying a Preconditioner
---------------------------

You can specify a preconditioner to improve the convergence of the iterative linear solution by setting the `precon` attribute. The
motivation for using a preconditioner is the observation that iterative methods have better convergence
properties if the linear system has a smaller condition number, so the goal of the preconditioner is to
improve the condition number in part or all of the Jacobian.

Here, we add a Gauss-Seidel preconditioner to the simple Sellar solution with Newton. Note that the number of
GMRES iterations is lower when using the preconditioner.

.. embed-test::
    openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_specify_precon

While the default preconditioning "side" is right-preconditioning, you can also use left-preconditioning provided that you choose
a "ksp_type" that supports it. Here we solve the same problem with left-preconditioning using the Richardson method and a `DirectSolver`.

.. embed-test::
    openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_specify_precon_left


.. tags:: Solver, LinearSolver
