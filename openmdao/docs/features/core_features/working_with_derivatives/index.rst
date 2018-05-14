
************************
Working with Derivatives
************************

Using Finite Difference or Complex Step
-----------------------------------------
.. toctree::
    :maxdepth: 1

    approximating_partials.rst
    approximating_totals.rst


Providing Analytic Partial Derivatives
----------------------------------------

.. toctree::
    :maxdepth: 1

    specifying_partials.rst
    sparse_partials.rst
    checking_partials.rst
    unit_testing_partials.rst


Computing Total Derivatives Across a Model
-------------------------------------------

.. toctree::
    :maxdepth: 1

    picking_mode.rst
    compute_totals.rst
    check_total_derivatives.rst


Reducing the Cost of Total Derivative Solves Using Advanced Features
---------------------------------------------------------------------

There are a number of special cases where a model has a particular structure that can be exploited to reduce the cost of linear solves used to compute total derivatives.
You can learn details of how to determine if your problem has the necessary structure to use one of these features, or how to restructure your problem to make use of them in the
:ref:`Theory Manual entry on how OpenMDAO computes total derivatives<theory_total_derivatives>`.

.. toctree::
    :maxdepth: 1

    assembled_jacobian.rst
    sparse_totals.rst
    simul_derivs.rst
    parallel_derivs.rst
    vectorized_derivs.rst
    linear_restart.rst
