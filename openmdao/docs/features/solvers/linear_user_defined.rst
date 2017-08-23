:orphan:

.. _lnuserdefined:

Linear Solver: LinearUserDefined
================================

This is a solver that let's you define custom method for performing a linear solve on a component. The default
method is named "solve_linear", but you can give it any name by passing in the function or method handle to
the "solve_function" attribute.

Here is a rather contrived example where an identity preconditioner is used by giving the compoennt's "mysolve"
method to a LinearUserDefined solver.

.. embed-test::
    openmdao.solvers.linear.tests.test_user_defined.TestUserDefinedSolver.test_feature

.. tags:: Solver, LinearSolver
