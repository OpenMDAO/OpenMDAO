****************************************************
Optimizing the Sellar Problem
****************************************************

In the previous tutorials we showed you how to define the Sellar model and run it directly.
Now lets see how we can optimize the Sellar problem to minimize the objective function.
Here is the mathematical problem formulation for the Sellar optimziation problem:

.. math::

    \begin{align}
    \text{min}: & \ \ \ & x_1^2 + z_2 + y_1 + e^-{y_2} \\
    \text{w.r.t.}: & \ \ \ &  x_1, z_1, z_2 \\
    \text{subject to}: & \ \ \ & \\
    & \ \ \ & 3.16 - y_1 <=0 \\
    & \ \ \ & y_2 - 24.0 <=0
    \end{align}

Remember that we built our Sellar model as follows:

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarMDA

All the variables we need to set up the optimization are there. So now we just need the run-script to execute the optimization.

.. embed-test::
    openmdao.test_suite.test_examples.test_sellar_opt.TestSellarOpt.test_sellar_opt


Controling the Solver Print Output
************************************
Notice the call to :code:`prob.set_solver_print()`,
which sets the solver output to level 0.
This is the semi-quiet setting where you only get notified if the solver failed to converge.
There are lots of ways to :ref:`configure the solver print <solver-options>` output in your model to suite your needs.


Why do have to set a linear solver?
***************************************
A linear solver is set for :code:`cycle` group as follows:

.. code::

    prob.model.cycle.linear_solver = DirectSolver()

In this run script, the solver was set after the :code:`setup` was called on the problem.
That was necessary because in the orignal definition of the group, the linear solver wasn't set.
We could have also set that up directly in the group's class definition as well:

.. code::

    cycle = self.add_subsystem('cycle', Group(), promotes=['*'])
    d1 = cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z', 'y2'], promotes_outputs=['y1'])
    d2 = cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z', 'y1'], promotes_outputs=['y2'])

    # Nonlinear Block Gauss Seidel is a gradient free solver
    cycle.nonlinear_solver = NonlinearBlockGS()
    cycle.linear_solver = DirectSolver()

Why do you have to set a linear solver at all? In OpenMDAO linear solvers are used to compute derivatives.
The :ref:`NonlinearBlockGS <nlbgs>` solver is gradient free and doesn't require a linear solver,
but we're doing optimization with a gradient based optimizer in this run script.
Any time you you're trying to optimize a model that has a non-linear solver converging multidisciplinary analysis
somewhere in your model then you will need to add an appropriate linear solver as well, so the framework can compute total derivatives.

OpenMDAO comes with a :ref:`number of different linear solvers <feature_linear_solvers>` that you can choose from.
Understanding which linear solver to pick requires considering a number of factors and is not a simple topic.
In many cases the :ref:`DirectSolver <directsolver>` is your best bet because it uses a lu factorization that is fairly robust.
For larger problems with 1000's of variables in them you could also try :ref:`ScipyIterativeSolver <scipyiterativesolver>` which uses Scipy's GMRES implementation.

.. note::

    Its important to realize that for this model, we set each individual component to approximate its own partial derivatives using FD.
    But that means that OpenMDAO is still responsible for computing the total derivatives across the whole model, which requires it to solve a linear system hence it needs the linear solver.
    If you don't want to use OpenMDAO's analytic derivatives capabilities you can tell the framework to finite-difference across the whole model instead.

