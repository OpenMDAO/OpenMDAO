.. _set-solvers:

************************************
Setting Nonlinear and Linear Solvers
************************************

A nonlinear solver, like :ref:`NonlinearBlockGS <openmdao.solvers.nonlinear.nonlinear_block_gs.py>` or :ref:`Newton <openmdao.solvers.nonlinear.newton.py>`,
is used to converge the nonlinear analysis. A nonlinear solver is needed whenever there is a cyclic dependency between components in your model.
It might also be needed if you have an :ref:`ImplicitComponent <openmdao.core.implicitcomponent.py>` in your model that expects the framework to handle its convergence.

Whenever you use a nonlinear solver on a :ref:`Group <openmdao.core.group.py>` or :ref:`Component <openmdao.core.component.py>`, if you're going to be working with analytic derivatives,
you will also need a linear solver.
A linear solver, like :ref:`LinearBlockGS <openmdao.solvers.linear.linear_block_gs.py>` or :ref:`DirectSolver <openmdao.solvers.linear.direct.py>`,
is used to solve the linear system that provides total derivatives across the model.

You can add nonlinear and linear solvers at any level of the model hierarchy,
letting you build a hierarchical solver setup to efficiently converge your model and solve for total derivatives across it.


Solvers for the Sellar Problem
------------------------------

The Sellar Problem has two components with a cyclic dependency, so the appropriate nonlinear solver is necessary.
We'll use the :ref:`Newton <openmdao.solvers.nonlinear.newton.py>` nonlinear solver,
which requires derivatives, so we'll also use the :ref:`Direct <openmdao.solvers.linear.direct.py>` linear solver.

.. embed-code::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_solver
    :layout: interleave

----

Some models have more complex coupling. There could be top-level cycles between groups as well as
lower-level groups that have cycles of their own. The openmdao.test_suite.components.double_sellar.DoubleSellar (TODO: Link to problem page)
is a simple example of this kind of model structure. In these problems, you might want to specify a more complex hierarchical solver structure for both nonlinear and linear solvers.

.. embed-code::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_subgroup_solvers
    :layout: interleave


.. note::
    Preconditioning for iterative linear solvers is a complex topic.
    The structure of the preconditioner should follow the model hierarchy itself,
    but developing an effective and efficient preconditioner is not trivial.
    If you're having trouble converging the linear solves with an iterative solver,
    you should try using the :ref:`Direct <openmdao.solvers.linear.direct.py>` solver instead.
    Before changing solvers, first verify that all your partials derivatives are correct with the `check_partials` method.


----

You can also specify solvers as part of the initialization of a Group

.. embed-code::
    openmdao.test_suite.components.double_sellar.DoubleSellar
    :layout: code

.. tags:: Solver
