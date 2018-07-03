.. _nlbroyden:

*************
BroydenSolver
*************

BroydenSolver is a quasi-Newton solver that implements Broyden's second method to solve for values of the model's states that
drive their residuals to zero. It does so by maintaning an approximation to the inverse of the Jacobian of the model or a subset
of the model. In some cases this can be more efficient than NewtonSolver because updating the approximated inverse Jacobian is
cheaper than solving the linear system. It may take more iterations because the search direction depends on an approximation,
but the iterations take fewer operations.

The BroydenSolver has two different modes of operation. It can operate on the entire model, and solve for every state in the containing
system and all subsystems. Alternatively, it can operate on a subset of the model, and only solve for a list of states that you provide.
The advanatage of full-model mode is that you don't have to worry about forgetting a state, particularly in large models where you might
not be familiar with every component or variable. The disadvantage is that you are computing the inverse of a larger matrix every time
you recalculate the inverse jacobian, though ideally you are not recomputing this very often. Operating on a subset of states is more
efficient in both the linear solve and the Broyden update, but you do run the risk of missing a state. The BroydenSolver will print a
warning if it finds any states in the model that aren't covered by a solver.


BroydenSolver Options
---------------------

.. embed-options::
    openmdao.solvers.nonlinear.broyden
    BroydenSolver
    options

The BroydenSolver also contains a slot for a linear solver and a slot for a linesearch.


BroydenSolver on a Full Model
-----------------------------

Here we show an example that uses the :ref:`electrical circuit model <using_balancecomp_tutorial>` from the
advanced guide. We have replaced the `NewtonSolver` with a `BroydenSolver`, and set the maximum number of iterations
to 20. We also assign a `DirectSolver` into the "linear_solver" slot on the `BroydenSolver`.  This is the linear solver
that will be used to assemble the Jacobian and compute its inverse. Since we don't specify any states in the `state_vars`
option, the BroydenSolver operates on the entire model. If you don't specify a linear_solver here, then the BroydenSolver
will use the one from the system.

.. note::
    In this mode, only the `DirectSolver` can be used as the linear_solver.

Depending on the values of some of the other options such as "converge_limit", "diverge_limit", and "max_converge_failures",
the Jacobian might be recalculated if convergence stalls, though this doesn't happen in the electrical circuit example.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_broyden.TestBryodenFeature.test_circuit_full
      :layout: code, output

BroydenSolver on a Subset of States
-----------------------------------

The `BroydenSolver` can also be used to solve for specific states. Here we consider the same circuit example, but instead
we specify the two voltages n1.V' and 'n2.V' as our "state_vars".  In this mode, we aren't limited to just using the
`DirectSolver`, and in this example we choose `LinearBlockGS` instead.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_broyden.TestBryodenFeature.test_circuit
      :layout: code, output

BroydenSolver for Models Without Derivatives
--------------------------------------------

The `BroydenSolver` can be used for models where you don't have any partial derivatives defined, and don't wish to use
finite difference to calculate them. This behavior is activated by setting the "compute_jacobian" option to False. Instead of calculating
an initial Jacobian, we start with an estimate that is just the identity matrix scaled by a tunable parameter
in the options called "alpha". As the `BroydenSolver` iterates, this estimate of the Jacobian is improved, and
for some problems, a solution can be reached that satisfies the residual equations.

In this example, we solve for the coupling variable in a version of the Sellar model that severs the cycle
and expresses the difference across the broken cycle as an implicit state, which the `BroydenSolver` will
solve.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_broyden.TestBryodenFeature.test_sellar
      :layout: code, output

BroydenSolver Option Examples
-----------------------------

There are a few additional options that give you more control over when and how often the Jacobian is recomputed.
The "diverge_limit" option allows you to define a limit to the ratio of current residual and the previous iteration's
residual above which the solution is considered to be diverging. If this limit is exceeded, then the Jacobian is
always recomputed on the next iteration. There is also a "converge_limit" that allows you similarly define a limit
above which the solution is considered to be non-converging. When this limit is exceeded, the Jacobian is not immediately
recomputed until the limit has been exceeded a number of consecutive times as defined by the "max_converge_failures"
option. The default value for "max_converge_failures" is 3, and the default "converge_limit" is 1.0. Exploring
these options can help you solve more quickly (or in some cases solve at all) some tougher problems.

Here, we take the same circuit example from above and specify a much lower "converge_limit" and "max_converge_failures"
to force recomputation of the Jacobian much more frequently. This results in a quicker convergence in terms of the
number of iterations, though keep in mind that solving for the derivatives adds computational cost.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_broyden.TestBryodenFeature.test_circuit_options
      :layout: code, output