.. _check-total-derivatives:

**************************
Checking Total Derivatives
**************************

If you want to check the analytic derivatives of your model (or just part of it) against finite difference or complex-step approximations, you can use :code:`check_totals()`. You should always converge your model
before calling this method.

.. note::
    You should probably **not** use this method until you've used :code:`check_partials()` to verify the
    partials for each component in your model. :code:`check_totals()` is a very blunt instrument, since it can only tell you that there is a problem, but will not give you much insight into which component or group is causing the problem.

.. automethod:: openmdao.core.problem.Problem.check_totals
    :noindex:

Examples
--------

You can check specific combinations of variables by specifying them manually:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_manual

----

Check the all the derivatives that the driver will need:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_from_driver

----

Display the results in a compact format:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_from_driver_compact

----

Use complex step instead of finite difference for a more accurate check. We also change to a larger
step size to trigger the nonlinear Gauss-Seidel solver to try to converge after the step.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_cs

----

Turn off standard output and just view the derivatives in the return:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_suppress

.. tags:: Derivatives
