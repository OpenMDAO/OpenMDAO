.. _check-total-derivatives:

**************************
Checking Total Derivatives
**************************

If you want to check the analytic derivatives of your model (or just part of it) against finite difference or complex-step approximations, you can use :code:`check_totals()`. You should always converge your model
before calling this method. By default, this method checks the derivatives of all of the driver responses (objectives, constraints) with respect to the des_vars, though you can also specify the
variables you want to check. Derivatives are computed
and compared in an unscaled form by default, but you can optionally request for them to be computed in scaled form using the `ref` and `ref0` that were declared when adding the
constraints, objectives, and des_vars.

.. note::
    You should probably **not** use this method until you've used :code:`check_partials()` to verify the
    partials for each component in your model. :code:`check_totals()` is a blunt instrument, since it can only tell you that there is a problem, but will not give you much insight into which component or group is causing the problem.

.. automethod:: openmdao.core.problem.Problem.check_totals
    :noindex:

Please check out the :ref:`Complex Step Guidelines <cs_guidelines>` for more information on how to use it on a model.

Examples
--------

You can check specific combinations of variables by specifying them manually:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_manual
    :layout: interleave

----

Check all the derivatives that the driver will need:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_from_driver
    :layout: interleave

----

Use the driver scaled values during the check:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_from_driver_scaled
    :layout: interleave

----

Display the results in a compact format:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_from_driver_compact
    :layout: interleave

----

Use complex step instead of finite difference for a more accurate check. We also tighten the tolerance
on the nonlinear Gauss-Seidel solver so that we get more accuracate converged values.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_cs
    :layout: interleave

Note that you need to set the argument "force_alloc_complex" to True when you call setup. This
makes sure that the vectors allocate space to pass complex numbers. All systems also have an
attribute "force_alloc_complex" that can be queried during `setup` in case you need to set up your
`Component` or `Group` differently to accomodate complex step.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_user_detect_forced
    :layout: code, output

----

Turn off standard output and just view the derivatives in the return:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_check_totals_suppress
    :layout: interleave

.. tags:: Derivatives
