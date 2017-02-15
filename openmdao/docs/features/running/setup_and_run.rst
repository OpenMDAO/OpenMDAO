:orphan:

.. _setup-and-run:

Setup and Run Your Model
=========================

After you have built up a model by defining variables and components, then organizing them into a hierarchy, and connecting them together, \
you need to call the :code:`setup()` method to have the framework do some initialization work in preparation for execution.
You can control some details of that initialization with the arguments that you pass into :code:`setup()`,
and it is important to note that you can not set or get any variable values nor run until **after** you call :code:`setup()`.

.. automethod:: openmdao.core.problem.Problem.setup
    :noindex:

Once :code:`setup()` is done, you can then execute your model in one of two ways:
    #. :code:`run_model()`
    #. :code:`run_driver()`

The first one, :code:`run_model()`, will execute one pass through your model.
The second one, :code:`run_driver()` executes the driver, running the optimization, DOE, etc. that you've set up.

Examples
---------

A basic :code:`setup()` using numpy vectors for the the framework data handling, and executing a single pass through the model.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_numpyvec_setup

----

To use any of the PETSc linear solvers and/or to run in parallel under MPI, the framework data handling should be done with the :code:`PETScVector`.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_petsc_setup

----

Set up a simple optimization problem and run it, call :code:`run_driver`.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_run_driver



Related Features
-------------------
:ref:`Set/Get Variables<set-and-get-variables>`, drivers
