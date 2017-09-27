In the last tutorial we built a run-script for an unconstrained optimization.
The extension to a constrained optimization is really strait forward, involving the addition of only a few lines of code.

********************************************************
Constrained Optimization of Paraboloid
********************************************************

If you want to add constraints to your optimization, the run script is nearly identical to the unconstrained case.
Here is the full run script with the constraint added.
Look over that first, then we'll examine the new lines of code a little more carefully below.

.. embed-test::
    openmdao.test_suite.test_examples.basic_opt_paraboloid.BasicOptParaboloid.test_constrainted


What has changed
--------------------

There are only two new lines of code added here, compared to the unconstrained case.
We used an :ref:`ExecComp <feature_exec_comp>` to create a new component with an output that we will constrain.
You could also an existing output from any component in your model if the value you want to constrain was already defined somewhere.
.. code::

    prob.model.add_subsystem('const', ExecComp('g = x + y'))

.. note ::

    :ref:`ExecComp <feature_exec_comp>` is a useful utility component provided in OpenMDAO's standard library that lets you define new calculations
    just by typing in the expression. It supports basic math operations, and even some of numpy's more advanced methods. You can work with
    both scalar and array data as well.

Once you have defined the necessary constraint function, you just have to add it to the problem formulation so the driver
knows to actually respect it. For this toy problem it turns out that the constrained optimum occurs when :math:`x = -y = 7.0`.,
so its actually possible to get the same answer using an equality constraint set to 0.

.. code::

    prob.model.add_constraint('const.g', lower=0, upper 10.)
    #prob.model.add_constraint('const.g', equals=0.)