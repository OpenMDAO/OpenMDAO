This tutorial will show you how to run unconstrained and constrained optimization on the really simple paraboloid model.

*******************************
Optimizing Paraboloid
*******************************

In OpenMDAO, optimizations are run by :code:`Drivers`.
In this tutorial we'll use :code:`ScipyOptimizer` which provides an interface
the various optimization algorithms in the scipy library. We'll use `SLSQP` which is a gradient based optimizer that
implements a sequential quadratic programming algorithm.

To start out we'll re-use the :code:`Paraboloid` component that we defined in the previous tutorial, and then we'll add
the necessary code to define and run the optimization.

Unconstrained Optimization
***************************

.. embed-test::
    openmdao.test_suite.test_examples.basic_opt_paraboloid.BaseicOptParaboloid.test_unconstrainted

The assignment of the driver is done by setting the :code:`driver` attribute of the problem.
Every problem has one, and only one, driver associated with it.
The default driver is just the base :code:`Driver` class itself, which will simply execute a single :code:`run_model()`
command.
Here we are changing the driver to an optimizer, and changing some settings on it.

.. code::

    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'

The we setup the problem formulation so the optimizer knows what to vary and what to optimize.
In these calls, you are always going to be specifying a specific variable. For :code:`add_design_var`
the variable will always be the output of an :code:`IndepVarComp`. For `add_objective` and `add_constraint`
the variable can be the output of any component (including an `IndepVarComp`).

.. code::

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f_xy')


Finally, we call :code:`setup` and then :code:`run_driver` to actually execute the model and use some print statements
to interogate the final values.



Constrained Optimization
***************************

If you want to add constraints to your optimization, the run script is nearly identical to the unconstrained case.

.. embed-test::
    openmdao.test_suite.test_examples.basic_opt_paraboloid.BaseicOptParaboloid.test_constrainted


There are only two new lines of code added here, compared to the unconstrained case.
Here we used an :code:`ExecComp` to create a new component whos output will be the value of our constraint equation,
but could also an existing output from any component in your model.
.. code::

    prob.model.add_subsystem('const', ExecComp('g = x + y'))

.. note ::

    :code:`ExecComp` is a useful utility component provided in OpenMDAO's standard library that lets you define new calculations
    just by typing in the expression. It supports basic math operations, and even some of numpy's more advanced methods. You can work with
    both scalar and array data as well. See the feature doc on ExecComp [TODO: add link] for some more examples of how to use it.

Once you have defined the necessary constraint function, you just have to add it to the problem formulation so the driver
knows to actually respect it. For this toy problem it turns out that the constrained optimum occurs when :math:`x = -y = 7.0`.,
so its actually possible to get the same answer using an equality constraint set to 0.

.. code::

    prob.model.add_constraint('const.g', lower=0, upper 10.)
    #prob.model.add_constraint('const.g', equals=0.)


