.. _nlbroyden:

*************
BroydenSolver
*************

BroydenSolver is a quasi-Newton solver that implements Broyden's first method (aka Broyden's good method) to solve
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