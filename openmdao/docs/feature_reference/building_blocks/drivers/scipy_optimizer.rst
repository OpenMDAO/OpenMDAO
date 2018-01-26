.. _scipyoptimizer:

*********************
ScipyOptimizer Driver
*********************

The ScipyOptimizer driver wraps the optimizers in `scipy.optimize.minimize`. In this example, we use the SLSQP
optimizer to find the minimum of the Paraboloid problem.

  .. embed-test::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_basic


ScipyOptimizeDriver Options
---------------------------

.. embed-options::
    openmdao.drivers.scipy_optimizer
    ScipyOptimizeDriver
    options

ScipyOptimizeDriver Option Examples
-----------------------------------

**optimizer**

  The "optimize" option lets you choose which optimizer to use. The ScipyOptimizer driver supports all
  of the optimizers in scipy.optimize except for 'dogleg' and 'trust-ncg'. Generally, the optimizers that
  you are most likely to use are "COBYLA" and "SLSQP", as these are the only ones that support constraints.
  Only SLSQP supports equality constraints, and SLSQP also uses gradients provide by OpenMDAO while COBYLA is
  gradient-free.

  .. embed-test::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_optimizer

**maxiter**

  The "maxiter" option is used to specify the maxinum number of major iterations before termination. It
  is generally a valid option across all of the available options.

  .. embed-test::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_maxiter

**tol**

  The "tol" option allows you to specify the tolerance for termination.

  .. embed-test::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_tol

.. tags:: Driver, optimization
