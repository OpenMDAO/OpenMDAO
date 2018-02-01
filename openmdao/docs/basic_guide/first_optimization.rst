This tutorial will show you how to setup and run an optimization using a component you've already defined.
The organization of this run script and its use of the :code:`Problem` class is the basis for executing all models in OpenMDAO.

*****************************************
Optimization of Paraboloid
*****************************************



To start out, we'll reuse the :code:`Paraboloid` component that we defined in the :ref:`previous tutorial <tutorial_paraboloid_analysis>`.
We'll add that component, along with an :ref:`IndepVarComp <comp-type-1-indepvarcomp>`, to construct our model
inside a :ref:`Problem <feature_run_your_model>`.
You've already used :code:`Problem` in the run script from the previous tutorial on the paraboloid analysis,
but we'll take a closer look now.

All analyses and optimizations in OpenMDAO are executed with an instance of the :code:`Problem` class.
This class serves as a container for your model and the driver you've chosen,
and provides methods for you to :ref:`run the model <run-model>` and :ref:`run the driver <setup-and-run>`.
It also provides a :ref:`interface for setting and getting variable values <set-and-get-variables>`.
Every problem has a single driver associated with it; similarly, every problem has a single model in it.

.. figure:: images/problem_diagram.png
   :align: center
   :width: 50%
   :alt: diagram of the problem structure


The Run Script
**************

.. embed-test::
    openmdao.test_suite.test_examples.basic_opt_paraboloid.BasicOptParaboloid.test_constrained
    :no-split:

Although we defined the :code:`Paraboloid` component in a :ref:`previous tutorial <tutorial_paraboloid_analysis>`, we wanted to add an additional equation to our model.
Since it was a very simple equation, we used the :ref:`ExecComp <feature_exec_comp>` to quickly add the new output to our model, so that we can constrain it.
Once you have defined the necessary output variable, you just have to add it to the problem formulation so the driver
knows to actually respect it. For this toy problem it turns out that the constrained optimum occurs when :math:`x = -y = 7.0`,
so it's actually possible to get the same answer using an equality constraint set to 0.
We included both options in the tutorial for your reference.

.. note ::

    :ref:`ExecComp <feature_exec_comp>` is a useful utility component provided in OpenMDAO's :ref:`standard library <feature_building_blocks>`
    that lets you define new calculations just by typing in the expression. It supports basic math operations, and even some of numpy's more
    advanced methods. It also supports both scalar and array data as well.

Setting a Driver
---------------------

Telling OpenMDAO to use a specific optimizer is done by setting the :code:`driver` attribute of the problem.
Here we'll use the :ref:`ScipyOptimizeDriver <scipyoptimizer>`, and tell it to use the *COBYLA* algorithm.

.. code::

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'

Defining the Design Variables and Objective
---------------------------------------------------------------

Next, we set up the problem formulation so that the optimizer knows what to vary and which objective to optimize.
In these calls, we are always going to be specifying a single variable. For :ref:`add_design_var <feature_add_design_var>`,
the variable will always be the output of an :ref:`IndepVarComp <comp-type-1-indepvarcomp>`.
For :ref:`add_objective <feature_add_objective>` and :ref:`add_constraint <feature_add_constraint>`
the variable can be the output of any component (including an :code:`IndepVarComp`).

.. code::

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f_xy')
        prob.model.add_constraint('const.g', lower=0, upper 10.)
        #prob.model.add_constraint('const.g', equals=0.)

.. note::

    Although these calls always point to a specific variable, that variable doesn't have to be a scalar value.
    See the feature docs for :ref:`adding design variables, objectives, and constraints <feature_adding_des_vars_obj_con>` for more details.


Finally, we call :ref:`setup <setup>`, and then :ref:`run_driver() <setup-and-run>` to actually execute the model, then we use some print statements
to interrogate the final values.






