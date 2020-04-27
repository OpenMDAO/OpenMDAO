.. _advanced_case_recording:

***************************
Advanced Recording Example
***************************

Below we demonstrate a more advanced use case of case recording including the four different objects
that a user can attach to. We will then show how to extract various data from the model and finally,
relate that an XDSM for illustrative purposes.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_advanced_example
    :layout: interleave


Below we have an XDSM to show the SellarMDA component equations and their inputs and outputs. Through
the different recorders we can access the different parts of the model. We'll take you through an
example of each object and relate it back to the diagram below.

.. image:: images/sellar_xdsm.jpg
    :width: 600

System
-------
First, we'll examine the `system` recorder. Suppose we want to know what the value of `y1` is going
into the objective function (obj_func). Using the `list_cases` method, we'll query the objective
function by passing in the string `root.obj_comp`. You could also access the discipline equations
by swapping out the subsystem `root.obj_comp` for `root.con_cmp1`. Next we use `get_case` to
inspect which variables are in the dictionary. Here we find that `x, y1, y2, z` are returned.
Since we originally sought find the value of `y1` going into the objective function, we'll loop
through the 14 cases to find what the value is in each case.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_system_recorder
    :layout: interleave

Solver
------

Similar to the `system` recorder, we can query the `solver` but in this case we will find the
calculated value of obj_func. You can also access the values of inputs to the equation with the
solver but in this case we'll focus on the obj_func value since we cannot get that information from
the `system` recorder.

We'll pass `'root.nonlinear_solver'` to the method list_cases, find how many cases there are and
arbitrarily pick number 3.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_recorder
    :layout: interleave

Driver
------
If we want to view the convergence of the model, the best place to find that is in the `Driver`. By
default, a recorder attached to a driver will record the design variables, constraints and
objectives, so we will print them for the model at the end of the optimization. We'll use the helper
methods like `get_objectives`, `get_design_vars`, `get_constraints` to return the info we're seeking.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_driver_recorder
    :layout: interleave

Problem
--------

A `Problem` recorder is best if you want to record an arbitrary case at a point in the model. In
our case, we have placed our point at the end of the model.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_problem_recorder
    :layout: interleave


Plotting Design Variables
-------------------------

When inspecting or debugging a model, it can be helpful to visualize the path of the design
variables to their final values. To do this, we can list the cases of the driver and plot the data
with respect to the iteration number.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_plot_des_vars
    :layout: interleave

.. image:: images/design_vars.jpg
    :width: 600