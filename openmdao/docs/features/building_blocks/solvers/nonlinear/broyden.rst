.. _nlbroyden:

*************
BroydenSolver
*************

BroydenSolver is a quasi-Newton solver that implements Broyden's second method (aka Broyden's good method) to solve
for states that drive the residuals to zero. Unlike Newton, BroydenSovler works on a subset of your model consisting
of just the implicit states that you specify. By default, Like Newton, Broyden's method steps in a direction defined by
the local jacobian of derivatives, but the key is that it doesn't need to be provided with that jacobian. You can optionally
provide one to improve the speed of convergence, however. BroydenSolver uses the linear solver that is slotted in
the containing system, though you can also add a different linear solver directly to the BroydenSolver.

BroydenSolver Options
---------------------

.. embed-options::
    openmdao.solvers.nonlinear.broyden
    BroydenSolver
    options

BroydenSolver for Models Without Derivatives
--------------------------------------------

The `BroydenSolver` can be used for models where you don't have derivatives defined, and don't wish to use
finite difference to calculate them. This is the default behavior of this solver. Instead of calculating
an initial Jacobian, we start with an estimate that is just the identity matrix scaled by a tunable parameter
in the options called "alpha". As the `BroydenSolver` iterates, this estimate of the Jacobian is improved, and
for some problems, a solution can be reached that satisfies the residual equations.

In this example, we solve for the coupling variable in a version of the Sellar model that severs the cycle
and expresses the difference across the broken cycle as an implicit state, which the `BroydenSolver` will
solve.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_broyden.TestBryodenFeature.test_sellar
      :layout: interleave

BroydenSolver for Models With Derivatives
-----------------------------------------

The `BroydenSolver` can also be used for models where you have derivative defined. The advantage of using a
`BroydenSolver` here is that you may be able to solve your model with a small number of linear solutions. In
some case, a good initial jacobian is all you need, and the cheap Broyden updates on subsequent iterations
will keep it converging towards the solution.

Keep in mind that, even if you didn't declare derivatives on all your components, it is still possible to use
finite difference (or possibly complex step) to compute a Jacobian for your model or submodel, as shown in
the feature doc for :ref:`approximating semi-total derivatives. <feature_declare_totals_approx>`

Here we show an example that uses the :ref:`electrical circuit model <using_balancecomp_tutorial>` from the
advanced guide. We have replaced the `NewtonSolver` with a `BroydenSolver`, and set the option "compute_jacobian"
to True so that it computes an initial Jacobian in the first iteration. Depending on the values of some of
the other options such as "converge_limit", "diverge_limit", and "max_converge_failures", the Jacobian
might be recalculated if convergence stalls, though this doesn't happen in the electrical circuit example.

  .. embed-code::
      openmdao.solvers.nonlinear.tests.test_broyden.TestBryodenFeature.test_circuit
      :layout: interleave


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
      :layout: interleave