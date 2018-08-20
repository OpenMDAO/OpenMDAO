.. _feature_compute_totals:

***************************
Computing Total Derivatives
***************************

:code:`Problem` has a method, :code:`compute_totals`, that allows you to compute the unscaled total derivative values
for a model.

If the model approximated its Jacobian, the method uses an approximation method.

.. automethod:: openmdao.core.problem.Problem.compute_totals
    :noindex:

Usage
-----

Here is a simple example of using :code:`compute_totals`:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_compute_totals
    :layout: interleave

By default, `compute_totals` returns the derivatives unscaled, but you can also request that they be scaled by
the driver scale values declared when the des_vars, objectives, or constraints are added:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_compute_totals_scaled
    :layout: interleave

.. tags:: Derivatives
