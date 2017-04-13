:orphan:

.. _check-total-derivatives:

Checking Total Derivatives
============================

If you want to check the analytic derivatives of your model (or just part of it) against finite-difference or complex-step approximations, you can use :code:`check_total_derivatives()`. You should always converge your model
before calling this method.

.. note::
    You should probably **not** use this method until you've used :code:`check_partial_derivs()` to verify the
    partials for each component in your model. :code:`check_total_derivatives()` is a very blunt instrument, since it can only tell you that there is a problem, but will not give you much insight into which component or group is causing the problem.

TODO: add this line ".. automethod:: openmdao.core.problem.Problem.check_total_derivatives
    :noindex:"

Examples
-----------

You can check specific combinations of variables by specifying them manually:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_total_derivatives_manual

----

Check the all the derivatives that the driver will need:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_total_derivatives_from_driver


Related Features
-----------------
check-partial-derivatives, :ref:`Set up Model<setup-and-run>`, :ref:`Run Model<setup-and-run>`
