:orphan:

.. _scipyiterativesolver:

Linear Solver: ScipyIterativeSolver
===================================

This is a serial solver, so it should never be used under MPI; use :ref:`PetscKSP <usr_openmdao.solvers.ln_petsc_ksp.py>`
instead.

Here, we calculate the total derivatives across the Sellar system.

.. embed-test::
    openmdao.solvers.tests.test_ln_scipy.TestScipyIterativeSolverFeature.test_specify_solver

Settings: maxiter
-----------------

This lets you specify the maximum number of Gauss Seidel iterations to apply. In this example, we
cut it back from the default (10) to 2 so that it terminates a few iterations earlier and doesn't
reach the specified absolute or relative tolerance.

.. embed-test::
    openmdao.solvers.tests.test_ln_scipy.TestScipyIterativeSolverFeature.test_feature_maxiter

Settings: atol
--------------

Here, we set the absolute tolerance to a looser value that will trigger an earlier termination. After
each iteration, the norm of the linear residuals is calculated by calling `apply_linear`. If this norm value is lower than the absolute
tolerance `atol`, the iteration will terminate.

.. embed-test::
    openmdao.solvers.tests.test_ln_scipy.TestScipyIterativeSolverFeature.test_feature_atol

Settings: rtol
--------------

The 'rtol' setting is not supported by Scipy GMRES.

.. tags:: Solver, LinearSolver