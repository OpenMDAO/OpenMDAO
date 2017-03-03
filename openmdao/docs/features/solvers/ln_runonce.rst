:orphan:

.. _lnrunonce:

Linear Solver: LNRunOnce
========================

The simplest linear solver in OpenMDAO is the LNRunOnce solver, which sequentially calls `apply_linear`
and `solve_linear` once on each subsystem. It is directly analogous to applying a single pass of the
chain rule to the whole system without any iteration at the top level. This linear solver can only be
used in systems where the following conditions are satisfied:

1. System does not contain a cycle.
2. System does not contain any implicit states.

However, subsystems can contain cycles or implicit states provided that they are using the appropriate
solver such as :ref:`ScipyIterativeSolver <usr_openmdao.solvers.ln_scipy.py>`

Here is an example of using an LNRunOnce solver to calculate the derivatives across a simple model with
the <Paraboloid> component.

.. embed-test::
    openmdao.solvers.tests.test_ln_runonce.TestLNRunOnceSolver.test_feature_solver

.. tags:: Solver, LinearSolver