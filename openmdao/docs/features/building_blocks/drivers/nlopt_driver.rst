.. _nlopt_driver:

***********
NLoptDriver
***********

NLoptDriver wraps the optimizers in `NLopt` and makes them accessible for OpenMDAO problems.
NLopt_ is a "free/open-source library for nonlinear optimization, providing a common interface for a number of different free optimization routines available online as well as original implementations of various other algorithms."
It includes methods for both local and global optimization and some of these methods use derivative information whereas others are derivative-free.

.. note::
    The NLopt package does not come included with the OpenMDAO installation. It is a separate optional package that can be installed via `pip install nlopt`.


In this simple example, we use the SLSQP algorithm to find the minimum of the Paraboloid problem.

  .. embed-code::
      openmdao.drivers.tests.test_nlopt_driver.TestNLoptDriverFeatures.test_feature_basic
      :layout: interleave


NLoptDriver Options
---------------------------

.. embed-options::
    openmdao.drivers.nlopt_driver
    NLoptDriver
    options

NLoptDriver Constructor
-------------------------------

The call signature for the `NLoptDriver` constructor is:

.. automethod:: openmdao.drivers.nlopt_driver.__init__
    :noindex:


NLoptDriver Option Examples
-----------------------------------

**optimizer**

  The "optimizer" option lets you choose which optimizer to use.
  NLoptDriver only supports some of the optimizers.

  Here we pass the optimizer option as a keyword argument.

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestNLoptDriverFeatures.test_feature_optimizer
      :layout: interleave

**maxiter**

  The "maxiter" option is used to specify the maximum number of major iterations before termination. It
  is generally a valid option across all of the available options.

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestNLoptDriverFeatures.test_feature_maxiter
      :layout: interleave

**tol**

  The "tol" option allows you to specify the tolerance for termination.

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestNLoptDriverFeatures.test_feature_tol
      :layout: interleave

.. tags:: Driver, Optimizer, Optimization


NLoptDriver Driver Specific Options
-------------------------------------------
Optimizers in `scipy.optimize.minimize` have optimizer specific options. To let the user specify values for these
options, OpenMDAO provides an option in the form of a dictionary named `opt_settings`. See the `scipy.optimize.minimize`
documentation for more information about the driver specific options that are available.

As an example, here is code using some `opt_settings` for the `shgo` optimizer:

  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestNLoptDriverFeatures.test_feature_shgo_rastrigin
      :layout: interleave

Notice that when using the `shgo` optimizer, setting the `opt_settings['maxiter']` to `None` overrides
`NLoptDriver`'s `options['maxiter']` value. It is not possible to set `options['maxiter']` to anything other
than an integer so the `opt_settings['maxiter']` option provides a way to set the `maxiter` value for the `shgo`
optimizer to `None`.

.. _mdolab: https://github.com/mdolab/pyoptsparse

.. _NLopt: https://nlopt.readthedocs.io/en/latest/