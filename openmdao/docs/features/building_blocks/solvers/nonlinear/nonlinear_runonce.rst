.. _nlrunonce:

****************
NonlinearRunOnce
****************

The simplest solver in OpenMDAO is NonlinearRunOnce, which executes the
system's components or subsystems sequentially. No iteration is performed by
this solver, so it can only be used in systems where the following conditions
are satisfied:

1. System does not contain a cycle, though subsystems may.
2. System does not contain any implicit states, though subsystems may.

Note that a subsystem may contain cycles or implicit states provided that it is
fitted with a solver that can handle them such as :ref:`NewtonSolver <openmdao.solvers.nonlinear.newton.py>`.

Here is an example of using NonlinearRunOnce for a simple model with the <Paraboloid> component.

.. embed-code::
    openmdao.solvers.nonlinear.tests.test_nonlinear_runonce.TestNonlinearRunOnceSolver.test_feature_solver
    :layout: interleave

NonlinearRunOnce Options
------------------------

.. embed-options::
    openmdao.solvers.nonlinear.nonlinear_runonce
    NonlinearRunOnce
    options

.. tags:: Solver, NonlinearSolver
