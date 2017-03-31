:orphan:

.. _lsboundscheck:

Line Search BoundsCheck
=======================

The `BoundsCheck` is a line search subsolver that can be specified in the `line_search` attribute
of a NewtonSolver.  BoundsCheck doesn't perform a tradtional line search, but is used to enforce
the upper and lower bounds on output variables. If a step along the Newton search gradient takes
it into a region that violates bounds, then it is backtracked to a point where there are no longer
any violations.

The following examples use a Newton solver on a component `ImplCompTwoStates` with an implicit output
'z' that has an upper bound of 2.5 and a lower bound of 1.5. Basic specification is shown here:

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_boundscheck_basic

Options
-------

- bound_enforcement: array

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_boundscheck_array

- bound_enforcement: scalar

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_boundscheck_scalar

- bound_enforcement: wall

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_boundscheck_wall

.. tags:: Linesearch