
****************
Simple GA Driver
****************

The SimpleGADriver is a driver based on a simple genetic algorithm implementation sourced from the lecture
notes for the class 2009 AAE550 taught by Prof. William A. Crossley. The implementation is pure Python.

The SimpleGADriver supports integer design variables, but it is not limited to them. You can use a genetic
algorithm with continuous design variables, but they must be discretized before being encoded into a binary string.
The SimpleGADriver will encode a continuous variable for you, but you must specify the number of bits for
the discretization via the 'bits' option. This number is the base-2 logarithm of the number of points you
want between the min and max value.  A higher value means more accuracy for this variable, but it also increases
the number of generations (and hence total evaluations) that will be required to find the minimum. If you do not
specify a value for bits for a continuous variable, then the variable is assumed to be integer, and encoded thusly.

The examples below show a mixed integer problem to illustrate usage of this driver with both integer and
discrete design variables.

.. embed-test::
    openmdao.drivers.tests.test_genetic_algorithm_driver.TestFeatureSimpleGA.test_basic

Optimizer Settings
==================

.. embed-options::
    openmdao.drivers.genetic_algorithm_driver
    SimpleGADriver
    options

You can change the number of generations to run the genetic algorithm by setting the "max_gen" option.

.. embed-test::
    openmdao.drivers.tests.test_genetic_algorithm_driver.TestFeatureSimpleGA.test_option_max_gen

You can change the population size by setting the "pop_size" option.

.. embed-test::
    openmdao.drivers.tests.test_genetic_algorithm_driver.TestFeatureSimpleGA.test_option_pop_size

If you have a model that doesn't contain any distributed components or parallel groups, then the model
evaluations for a new generation can be performed in parallel by turning on the "parallel" option:

.. embed-test::
    openmdao.drivers.tests.test_genetic_algorithm_driver.MPIFeatureTests.test_option_parallel