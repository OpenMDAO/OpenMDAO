.. _lnrunonce:

*************
LinearRunOnce
*************

The simplest linear solver in OpenMDAO is LinearRunOnce, which sequentially calls `apply_linear`
and `solve_linear` once on each subsystem. It is directly analogous to applying a single pass of the
chain rule to the whole system without any iteration at the top level. This linear solver can only be
used in systems where the following conditions are satisfied:

1. System does not contain a cycle, though subsystems may.
2. System does not contain any implicit states, though subsystems may.

However, subsystems can contain cycles or implicit states, provided that they are using the appropriate
solver such as :ref:`ScipyKrylov <openmdao.solvers.linear.scipy_iter_solver.py>`

Here is an example of using LinearRunOnce to calculate the derivatives across a simple model with
the `Paraboloid` component.

.. embed-test::
    openmdao.solvers.linear.tests.test_linear_runonce.TestLinearRunOnceSolver.test_feature_solver


LinearRunOnce Options
---------------------

.. embed-options::
    openmdao.solvers.linear.linear_runonce
    LinearRunOnce
    options

.. tags:: Solver, LinearSolver