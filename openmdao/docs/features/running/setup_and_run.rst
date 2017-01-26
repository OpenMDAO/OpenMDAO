Setup and run your model
=========================

After you have built up a model by defining variables and components then organizing them into a hierarchy and connecting them together, \
you need to call the :code:`setup()` method to have the framework do some initialization work in preparation for execution.
You can control some details of that initialization by arguments you pass into setup,
and it is important to note is that you can not set or get any variable values or run until **after** you call :code:`setup()`.

Once setup is done, you can then execute your model in one of two ways:
    #. :code:`run_model()`
    #. :code:`run_driver()`

The first one will execute one pass through your model.
The second one executes the  <openmdao.core.driver.Driver> running the optimization, DOE, etc. that you've set up.

.. embed-autodoc:
    openmdao.core.problem.setup


Examples
---------

A basic setup using numpy vectors for the the framework data handling, and executing a single pass through the mode.

.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_numpyvec_setup

----

To use any of the PETSc linear solvers and/or to run in parallel under MPI, the framework data handling should be done with the :code:`PETScVector`.

.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_petsc_setup

----

Setup a simple optimization problem and run it, call :code:`run_driver`.

.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_run_driver



Related Features
-------------------
set_get, drivers

[TODO: auto-link to the appropriate top other feature docs!]
