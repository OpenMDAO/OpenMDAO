.. _debugging-drivers:

*********************
Driver Debug Printing
*********************

When working with a model, it may sometimes be helpful to print out the design variables, constraints, and
objectives as the :code:`Driver` iterates. OpenMDAO provides options on the :code:`Driver` to let you do that.

Usage
-----

This example shows how to use the :code:`Driver` debug printing options. The :code:`debug_print` option is a list of strings.
The only valid strings are 'desvars', 'ln_cons', 'nl_cons', and 'objs'. Note that the values for the design variables
printed are unscaled, physical values.


  .. embed-code::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_debug_print_option
      :layout: interleave
