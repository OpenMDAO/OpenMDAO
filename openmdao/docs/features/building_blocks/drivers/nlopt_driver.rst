.. _nlopt_driver:

***********
NLoptDriver
***********

NLoptDriver wraps the optimizers in `NLopt` and makes them accessible for OpenMDAO problems.
NLopt_ is a "free/open-source library for nonlinear optimization, providing a common interface for a number of different free optimization routines available online as well as original implementations of various other algorithms."
It includes `methods for both local and global optimization <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_ and some of these methods use derivative information whereas others are derivative-free.

Depending on your optimization problem formulation, one method might be more advantageous to use over others.
In general, there is no single best optimization algorithm.
Please see the NLopt documentation for more information on how the methods are implemented and best used. 

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

  Here we pass the optimizer option as a keyword argument, but it could be passed as an option after we instantiate the driver as well.
  The other examples show options being set after instantiation.

  .. embed-code::
      openmdao.drivers.tests.test_nlopt_driver.TestNLoptDriverFeatures.test_feature_optimizer
      :layout: interleave

**maxiter**

  The "maxiter" option is used to specify the maximum number of major iterations before termination. It
  is generally a valid option across all of the available options.

  .. embed-code::
      openmdao.drivers.tests.test_nlopt_driver.TestNLoptDriverFeatures.test_feature_maxiter
      :layout: interleave

**tol**

  The "tol" option allows you to specify the tolerance for termination.

  .. embed-code::
      openmdao.drivers.tests.test_nlopt_driver.TestNLoptDriverFeatures.test_feature_tol
      :layout: interleave

      
.. tags:: Driver, Optimizer, Optimization

.. _mdolab: https://github.com/mdolab/pyoptsparse

.. _NLopt: https://nlopt.readthedocs.io/en/latest/