Setting nonlinear and linear solvers
=====================================

A nonlinear solver, like <openmdao.solvers.nl_bgs.NonlinearBlockGS> or <openmdao.solvers.nl_newton.Newton>,
is used to converge the nonlinear analysis. A nonlinear solver is needed whenever this is either a cyclic dependency between components in your model.
It might also be needed if you have an <openmdao.core.implicitcomponent.ImplicitComponent> in your model that expects the framework to handle its convergence.

Whenever you use a nonlinear solver on a <openmdao.core.group.Group> or <openmdao.core.component.Component>, if you're going to be working with analytic derivatives,
you will also need a linear solver.
A linear solver, like <openmdao.solvers.ln_bgs.LinearBlockGS> or <openmdao.solvers.ln_direct.DirectSolver>,
is used to solve the linear system that provides total derivatives across the model.

You can add nonlinear and linear solvers at any level of the model hierarchy,
letting you build a hierarchical solver setup to efficiently converge your model and solve for total derivatives across it.


Solvers for the Sellar problem
----------------------------------

The Sellar (link to sellar problem page) problem has two components with a cyclic dependency,
so a a nonlinear solver is necessary.
We'll use the <openmdao.solvers.nl_newton.Newton> nonlinear solver,
which requires derivatives so a we'll also use the <openmdao.solvers.ln_direct.Direct) linear solver

.. embed-test::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_solver

The answers given are the expected solution to the problem given the default initial conditions for x and z.

----

Next, let's show an example of setting nonlinear solvers at different levels in your model hieararchy.  We take our current Sellar model and place it
into a group called SellarSubGroup. Our new model contains two SellarSubGroups, where we connect 'y1' from each group to the input 'x' of the other
group. This forms another cycle that needs to be converged.  For this cycle, we will use the `NonlinearBlockGS` solver, which implements Gauss-Seidel
(or Fixed Point Iteration). In the following code, the Gauss-Seidel solver is in the top group ("model"), while the Newton solver is in each of the
Sellar subgroups.

.. embed-python-code::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_subgroup_solvers


Sub-solvers for other solvers
-------------------------------

Some nonlinear solvers need other solvers to help them converge.
For example <openmdao.solvers.nl_newton.Newton> needs a linear solver to solver the newton update equation and can optionally use a line-search algorithm for globalization.

----

Similarly, some linear solvers can use sub-solvers as pre-conditioners to help them solve for derivatives more efficiently.

.. note::
    Preconditioning for iterative linear solvers is complex topic.
    The structure of the preconditioner should follow the model hierarchy itself,
    but developing an effective and efficient preconditioner is not trivial.
    If you're having trouble getting solving for total derivatives because the linear solver isn't converging
    (and you've already verified that all your partials derivatives are correct with the check_partial_derivatives method!) then you should try using the
    <openmdao.solvers.ln_direct.DirectSolver>.


