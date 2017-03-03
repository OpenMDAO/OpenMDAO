:orphan:

.. _directsolver:

Linear Solver: DirectSolver
===========================

The DirectSolver is a linear solver that assembles the system Jacobian and solves the linear
system with LU factorization and back substitution. It can handle any system topology. Since it
assembles a global Jacobian for all of its subsystems, any linear solver that is assigned in
any of its subsystems does not participate in this calculation (though they may be used in other
ways such as in subystem Newton solves.)

Here, we calculate the total derivatives across the Sellar system.

.. embed-test::
    openmdao.solvers.tests.test_ln_direct.TestDirectSolverFeature.test_specify_solver

.. tags:: Solver, LinearSolver