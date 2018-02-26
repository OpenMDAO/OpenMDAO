.. _run-model:

**************
Run Your Model
**************

Once :code:`setup()` is done, you can then execute your model with :code:`run_model()`

:code:`run_model()` will execute one pass through your model.


Examples
---------

A basic :code:`setup()` using numpy vectors for the framework data handling, and executing a single pass through the model.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_numpyvec_setup
    :layout: interleave

----

To use any of the PETSc linear solvers and/or to run in parallel under MPI, the framework data handling should be done with the :code:`PETScVector`.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_petsc_setup
    :layout: interleave
