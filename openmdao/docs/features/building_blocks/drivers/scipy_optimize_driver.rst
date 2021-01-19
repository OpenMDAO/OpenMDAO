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

ScipyOptimizeDriver Constructor
-------------------------------

The call signature for the `ScipyOptimizeDriver` constructor is:

.. automethod:: openmdao.drivers.scipy_optimizer.ScipyOptimizeDriver.__init__
    :noindex:


ScipyOptimizeDriver Option Examples
-----------------------------------

**optimizer**

  The "optimizer" option lets you choose which optimizer to use. ScipyOptimizeDriver supports all
  of the optimizers in scipy.optimize except for 'dogleg' and 'trust-ncg'. Generally, the optimizers that
  you are most likely to use are "COBYLA" and "SLSQP", as these are the only ones that support constraints.
  Only SLSQP supports equality constraints. SLSQP also uses gradients provided by OpenMDAO, while COBYLA is
  gradient-free.  Also, SLSQP supports both equality and inequality constraints, but COBYLA only supports
  inequality constraints.

  Here we pass the optimizer option as a keyword argument.

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


ScipyOptimizeDriver Driver Specific Options
-------------------------------------------
Optimizers in `scipy.optimize.minimize` have optimizer specific options. To let the user specify values for these
options, OpenMDAO provides an option in the form of a dictionary named `opt_settings`. See the `scipy.optimize.minimize`
documentation for more information about the driver specific options that are available.

As an example, here is code using some `opt_settings` for the `shgo` optimizer:

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_shgo_rastrigin
      :layout: interleave

Notice that when using the `shgo` optimizer, setting the `opt_settings['maxiter']` to `None` overrides
`ScipyOptimizeDriver`'s `options['maxiter']` value. It is not possible to set `options['maxiter']` to anything other
than an integer so the `opt_settings['maxiter']` option provides a way to set the `maxiter` value for the `shgo`
optimizer to `None`.
