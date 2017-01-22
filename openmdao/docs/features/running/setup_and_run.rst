Setup and run your model
=========================

After you have built up a model by defining variables and components then organizing them into a hierarchy and connecting them together, \
you need to call the :code:`setup()` method to have the framework do some initialization work in preparation for execution.
You can control some details of that initialization by arguments you pass into setup,
and it is important to note is that you can not set or get any variable values or run until **after** you call :code:`setup()`.

.. embed-autodoc:
    openmdao.core.problem.setup


Examples
---------

Basic setup, using numpy vectors for data passing:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_numpyvec_setup


To use any of the PETSc linear solvers or to run in parallel under MPI:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_petsc_setup












