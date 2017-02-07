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

----

Some models have more complex coupling. There could be top level cycles between groups as well as
lower level groups that have cycles of their own. The openmdao.test_suite.components.double_sellar.DoubleSellar (TODO: Link to problem page) is a simple example of this kind of model structure. In these problems, you might want to specify a more complex hierarchical solver structure for both nonlinear and linear solvers.

.. embed-test::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_subgroup_solvers


.. note::
    Preconditioning for iterative linear solvers is complex topic.
    The structure of the preconditioner should follow the model hierarchy itself,
    but developing an effective and efficient preconditioner is not trivial.
    If you're having trouble converging the linear solves with an iterative solver,
    you should try using the <openmdao.solvers.ln_direct.DirectSolver> instead.
    But first, verify that all your partials derivatives are correct with the check_partial_derivatives method.


----

You can also specify solvers as part of the initialization of a Group

.. embed-code::
    openmdao.test_suite.components.double_sellar.DoubleSellar

