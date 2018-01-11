.. _debugging-drivers:

*********************
Driver Debug Printing
*********************

When working with a model, it may sometimes be helpful to print out the design variables, constraints, and
objectives as the `Driver` iterates. OpenMDAO provides options on the `Driver` to let you do that.

Usage
-----

This example shows how to use the `Driver` debug printing options.


  .. embed-test::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizerFeatures.test_debug_print_option


Driver Debug Printing Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a summary of the `Driver` `debug_print` options.

.. embed-options::
    openmdao.core.driver
    Driver
    debug_print




