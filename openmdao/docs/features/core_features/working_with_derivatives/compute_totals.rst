.. _feature_compute_totals:

***************************
Computing Total Derivatives
***************************

:code:`Problem` has a method, :code:`compute_totals`, that allows you to compute the total derivative values
for a model.

If the model approximated its Jacobian, the method uses an approximation method.

.. automethod:: openmdao.core.problem.Problem.compute_totals
    :noindex:

Usage
-----

Here is a simple example of using :code:`compute_totals`:

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_compute_totals

.. tags:: Derivatives
