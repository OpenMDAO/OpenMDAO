.. _debugging-drivers:

*********************
Driver Debug Printing
*********************

When working with a model, it may sometimes be helpful to print out the design variables, constraints, and
objectives as the `Driver` iterates. OpenMDAO provides options on the `Driver` to let you do that.

Usage
-----

This example shows how to use the `Driver` debug printing options. The `debug_print` option is a list of strings.
The only valid strings are 'desvars','ln_cons','nl_cons',and 'objs'.


  .. embed-test::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizerFeatures.test_debug_print_option


