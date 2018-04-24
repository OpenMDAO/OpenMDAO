.. _scipy_optimize_driver:

*******************
ScipyOptimizeDriver
*******************

ScipyOptimizeDriver wraps the optimizers in `scipy.optimize.minimize`. In this example, we use the SLSQP
optimizer to find the minimum of the Paraboloid problem.

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_basic
      :layout: interleave


ScipyOptimizeDriver Options
---------------------------

.. embed-options::
    openmdao.drivers.scipy_optimizer
    ScipyOptimizeDriver
    options

ScipyOptimizeDriver Option Examples
-----------------------------------

**optimizer**

  The "optimizer" option lets you choose which optimizer to use. ScipyOptimizeDriver supports all
  of the optimizers in scipy.optimize except for 'dogleg' and 'trust-ncg'. Generally, the optimizers that
  you are most likely to use are "COBYLA" and "SLSQP", as these are the only ones that support constraints.
  Only SLSQP supports equality constraints. SLSQP also uses gradients provided by OpenMDAO, while COBYLA is
  gradient-free.

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_optimizer
      :layout: interleave

**maxiter**

  The "maxiter" option is used to specify the maximum number of major iterations before termination. It
  is generally a valid option across all of the available options.

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_maxiter
      :layout: interleave

**tol**

  The "tol" option allows you to specify the tolerance for termination.

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_tol
      :layout: interleave

.. tags:: Driver, Optimizer, Optimization
