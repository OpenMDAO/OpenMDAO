.. _debugging-drivers:

**************************
Driver Debug Print Options
**************************

When working with a model, it may sometimes be helpful to print out the design variables, constraints, and
objectives as the `Driver` iterates. OpenMDAO provides options on the `Driver` to let you do that.

Usage
-----

*Debug Print Options*
~~~~~~~~~~~~~~~~~~~~~


  .. embed-test::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizerFeatures.test_debug_print_option


Driver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.driver
    Driver
    debug_print




