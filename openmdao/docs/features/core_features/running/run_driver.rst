.. _setup-and-run:

************
Run a Driver
************

Once :code:`setup()` is done, you can then run the optimization with :code:`run_driver()`.

:code:`run_driver()` executes the driver, running the optimization, DOE, etc. that you've set up.

Examples
--------

Set up a simple optimization problem and run it, by calling :code:`run_driver`.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_run_driver
    :layout: interleave
