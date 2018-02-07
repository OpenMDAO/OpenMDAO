.. _feature_compute_totals:

***************************
Computing Total Derivatives
***************************

:code:`Problem` has a method, :code:`compute_totals`, that computes derivatives of desired quantities with respect to
desired inputs.

This method allows you to compute the derivative values for the model.

.. automethod:: openmdao.core.problem.Problem.compute_totals
    :noindex:

Usage
-----

Here is a simple example of using :code:`compute_totals`:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_compute_totals

.. tags:: Derivatives
