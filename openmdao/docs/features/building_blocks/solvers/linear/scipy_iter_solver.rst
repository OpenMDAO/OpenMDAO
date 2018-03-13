.. _scipyiterativesolver:

***********
ScipyKrylov
***********

ScipyKrylov is an iterative linear solver that wraps the methods found in `scipy.sparse.linalg`.
The default method is "gmres", or the Generalized Minimal RESidual method. Support for other
`scipy.sparse.linalg` solvers will be added over time. This linear solver is capable of handling any
system topology very effectively. It also solves all subsystems below it in the hierarchy, so
assigning different solvers to subsystems will have no effect on the solution at this level.

This is a serial solver, so it should never be used under MPI; use :ref:`PETScKrylov <openmdao.solvers.linear.petsc_ksp.py>`
instead.

Here, we calculate the total derivatives across the Sellar system.

.. embed-code::
    openmdao.solvers.linear.tests.test_scipy_iter_solver.TestScipyKrylovFeature.test_specify_solver
    :layout: interleave

ScipyKrylov Options
-------------------

.. embed-options::
    openmdao.solvers.linear.scipy_iter_solver
    ScipyKrylov
    options

ScipyKrylov Option Examples
---------------------------

**maxiter**

  `maxiter` lets you specify the maximum number of GMRES iterations to apply. The default maximum is 1000, which
  is much higher than the other linear solvers because each multiplication by the system Jacobian is considered
  to be an iteration. You may have to decrease this value if you have a coupled system that is converging
  very slowly. (Of course, in such a case, it may be better to add a preconditioner.)  Alternatively, you
  may have to raise it if you have an extremely large number of components in your system (a 1000-component
  ring would need 1000 iterations just to make it around once.)

  This example shows what happens if you set maxiter too low (the derivatives should be nonzero, but it stops too
  soon.)

  .. embed-code::
      openmdao.solvers.linear.tests.test_scipy_iter_solver.TestScipyKrylovFeature.test_feature_maxiter
      :layout: interleave

**atol**

  Here, we set the absolute tolerance to a much tighter value (default is 1.0e-12) to show what happens. In
  practice, the tolerance serves a dual role in GMRES. In addition to being a termination criteria, the tolerance
  also defines what GMRES considers to be tiny. Tiny numbers are replaced by zero when the argument vector is
  normalized at the start of each new matrix-vector product. The end result here is that we iterate longer to get
  a marginally better answer.

  You may need to adjust this setting if you have abnormally large or small values in your global Jacobian.

  .. embed-code::
      openmdao.solvers.linear.tests.test_scipy_iter_solver.TestScipyKrylovFeature.test_feature_atol
      :layout: interleave

**rtol**

  The 'rtol' setting is not supported by Scipy GMRES.

Specifying a Preconditioner
---------------------------

You can specify a preconditioner to improve the convergence of the iterative linear solution by setting the `precon` attribute. The
motivation for using a preconditioner is the observation that iterative methods have better convergence
properties if the linear system has a smaller condition number, so the goal of the preconditioner is to
improve the condition number in part or all of the Jacobian.

Here, we add a Gauss-Seidel preconditioner to the simple Sellar solution with Newton. Note that the number of
GMRES iterations is lower when using the preconditioner.

.. embed-code::
    openmdao.solvers.linear.tests.test_scipy_iter_solver.TestScipyKrylovFeature.test_specify_precon
    :layout: interleave

**A note on nesting ScipyKrylov under a preconditoner:** The underlying GMRES module is not
re-entrant, so it cannot be called as a new instance while it is running. If you need to use gmres under
gmres in a preconditioner stack, you should use :ref:`PETScKrylov <openmdao.solvers.linear.petsc_ksp.py>` at
one (ore more) of the levels.

.. tags:: Solver, LinearSolver
