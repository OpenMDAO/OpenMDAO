.. _feature_amijo_goldstein:

****************
Armijo-Goldstein
****************

ArmijoGoldsteinLS checks bounds and backtracks to a point that satisfies them. From there,
further backtracking is performed, until the termination criteria are satisfied.
The main termination criteria is the Armijo-Goldstein condition, which checks for a sufficient
decrease from the initial point by measuring the slope. There is also an iteration maximum.

Armijo-Goldstein Options
------------------------

.. embed-options::
    openmdao.solvers.linesearch.backtracking
    ArmijoGoldsteinLS
    options


Armijo-Goldstein Option Examples
--------------------------------

**bound_enforcement**

ArmijoGoldsteinLS includes the `bound_enforcement` option in its options dictionary. This option has a dual role:

1. Behavior of the non-bounded variables when the bounded ones are capped.
2. Direction of the further backtracking.

There are three different acceptable values for bounds-enforcement schemes available in this option.

With "vector" bounds enforcement, the solution in the output vector is pulled back to a point where none of the
variables violate any upper or lower bounds. Further backtracking continues along the Newton gradient direction vector back towards the
initial point.

.. image:: BT1.jpg

With "scalar" bounds enforcement, only the variables that violate their bounds are pulled back to feasible values; the
remaining values are kept at the Newton-stepped point. This changes the direction of the backtracking vector so that
it still moves in the direction of the initial point.

.. image:: BT2.jpg

With "wall" bounds enforcement, only the variables that violate their bounds are pulled back to feasible values; the
remaining values are kept at the Newton-stepped point. Further backtracking only occurs in the direction of the non-violating
variables, so that it will move along the wall.

.. image:: BT3.jpg

Here are examples of each acceptable value for the **bound_enforcement** option:

- bound_enforcement: vector

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "vector", the output vector is rolled back along the computed gradient until
  it reaches a point where the earliest bound violation occurred. The backtracking continues along the original
  computed gradient.

.. embed-test::
    openmdao.solvers.linesearch.tests.test_backtracking.TestFeatureLineSearch.test_feature_boundscheck_vector

- bound_enforcement: scalar

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "scaler", then the only indices in the output vector that are rolled back
  are the ones that violate their upper or lower bounds. The backtracking continues along the modified gradient.

.. embed-test::
    openmdao.solvers.linesearch.tests.test_backtracking.TestFeatureLineSearch.test_feature_boundscheck_scalar

- bound_enforcement: wall

  The `bound_enforcement` option in the options dictionary is used to specify how the output bounds
  are enforced. When this is set to "wall", then the only indices in the output vector that are rolled back
  are the ones that violate their upper or lower bounds. The backtracking continues along a modified gradient
  direction that follows the boundary of the violated output bounds.

.. embed-test::
    openmdao.solvers.linesearch.tests.test_backtracking.TestFeatureLineSearch.test_feature_boundscheck_wall

**maxiter**

  The "maxiter" option is a termination criteria that specifies the maximum number of backtracking steps to allow.

**alpha**

  The "alpha" option is used to specify the initial length of the Newton step. Since NewtonSolver assumes a
  stepsize of 1.0, this value usually shouldn't be changed.

**rho**

  The "rho" option controls how far to backtrack in each successive backtracking step. It is applied as a multiplier to
  the step, so a higher value (approaching 1.0) is a very small step, while a low value takes you close to the initial
  point. The default value is 0.5.

**c**

  In the `ArmijoGoldsteinLS`, the "c" option is a multiplier on the slope check. Setting it to a smaller value means a more
  gentle slope will satisfy the condition and terminate.

**print_bound_enforce**

  When the "print_bound_enforce" option is set to True, the line-search will print the name and values of any variables
  that exceeded their lower or upper bounds and were drawn back during bounds enforcement.

.. embed-test::
    openmdao.solvers.linesearch.tests.test_backtracking.TestFeatureLineSearch.test_feature_print_bound_enforce

- retry_on_analysis_error

  By default, the ArmijoGoldsteinLS linesearch will backtrack if the model raises an AnalysisError, which can happen if
  the component explicitly raises it, or a subsolver hits its iteration limit with the 'err_on_maxiter' option set to True.
  If you would rather terminate on an AnalysisError, you can set this option to False.

.. tags:: linesearch, backtracking
