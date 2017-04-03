:orphan:

.. _lsbacktracking:

Line Search and Backtracking
============================

Backtracking line searches are subsolvers that can be specified in the `line_search` attribute
of a NewtonSolver and are used to pull back to a reasonable point when a Newton step goes to far. This
can occur when a step causes output variables to exceed their specified lower and upper bounds. It can
also happen in more complicated problems where a full Newton step happens to take you well past the nonlinear solution,
even to an area where the residual norm is worse than the initial point. Specifying a line_search can
help alleviate these problems and improve robustness of your Newton solve.

There are three different backtracking line-search algorithms in OpenMDAO:

BoundsCheck
  Only checks bounds and backtracks to point that satisfies them.

BacktrackingLineSearch
  Checks bounds and backtracks to point that satisfies them. From there, further backtracking is performed until the termination criteria are satisfied; these
  criteria include a relative and aboslute tolerance and an iteration maximum.

ArmijoGoldstein
  Checks bounds and backtracks to point that satisfies them. From there, further backtracking is performed until the termination criteria are satisfied.
  The main termination criteria is the AmijoGoldstein condition, which checks for a sufficient decrease from the initial point by measuring the
  slope. There is also an iteration maximum.

The following examples use a Newton solver on a component `ImplCompTwoStates` with an implicit output
'z' that has an upper bound of 2.5 and a lower bound of 1.5. This example shows how to specify a line search
(which in this case is the `BacktrackingLineSearch`.):

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_backtrack_basic

Bound Enforcement
-----------------

All of the backtracking subsolvers include the `bound_enforcement` option in the options dictionary. This option has a dual role:

1. Behavior of the the non-bounded variables when the bounded ones are capped.
2. Direction of the further backtracking.

Options
-------

- bound_enforcement: vector

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "vector", the output vector is rolled back along the computed gradient until
  it reaches a point where the earliest bound violation occured. The backtracking continues along the original
  computed gradient.

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_backtrack_vector

- bound_enforcement: scalar

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "scaler", then the only indices in the output vector that are rolled back
  are the ones that violate their upper or lower bounds. The backtracking continues along the modified gradient.

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_backtrack_scalar

- bound_enforcement: wall

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "wall", then the only indices in the output vector that are rolled back
  are the ones that violate their upper or lower bounds. The backtracking continues along a modified gradient
  direction that follows the boundary of the violated output bounds.

.. embed-test::
    openmdao.solvers.tests.test_ls_backtracking.TestFeatureBacktrackingLineSearch.test_feature_backtrack_wall

.. tags:: Linesearch