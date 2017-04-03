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

- bound_enforcement: vector

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "vector", the output vector is rolled back along the computed gradient until
  it reaches a point where the earliest bound violation occured.

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_boundscheck_vector

- bound_enforcement: scalar

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "scaler", then the only indices in the output vector that are rolled back
  are the ones that violate their upper or lower bounds.

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_boundscheck_scalar

- bound_enforcement: wall

  Setting `bound_enforement`` to "wall" is equivalent to setting it to "scaler".

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_boundscheck_wall

.. tags:: Linesearch