.. _directsolver:

************
DirectSolver
************

DirectSolver is a linear solver that assembles the system Jacobian and solves the linear
system with LU factorization and back substitution. It can handle any system topology. Since it
assembles a global Jacobian for all of its subsystems, any linear solver that is assigned in
any of its subsystems does not participate in this calculation (though they may be used in other
ways such as in subsystem Newton solves.)

Here we calculate the total derivatives of the Sellar system objective with respect to the design
variable 'z'.

.. embed-code::
    openmdao.solvers.linear.tests.test_direct_solver.TestDirectSolverFeature.test_specify_solver
    :layout: interleave


DirectSolver Options
--------------------

.. embed-options::
    openmdao.solvers.linear.direct
    DirectSolver
    options

.. tags:: Solver, LinearSolver
