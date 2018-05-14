
**************
SimpleGADriver
**************

.. note::
    SimpleGADriver is based on a simple genetic algorithm implementation sourced from the lecture
    notes for the class 2009 AAE550 taught by Prof. William A. Crossley at Purdue University.

This genetic algorithm optimizer supports integer and continuous variables.
It uses a binary encoding scheme to encode any continuous variables into a user-definable number of bits.
The number of bits you choose should be equal to the base-2 logarithm of the number of discrete values you
want between the min and max value.  A higher value means more accuracy for this variable, but it also increases
the number of generations (and hence total evaluations) that will be required to find the minimum. If you do not
specify a value for bits for a continuous variable, then the variable is assumed to be integer, and encoded as such.

The examples below show a mixed-integer problem to illustrate usage of this driver with both integer and
discrete design variables.

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.TestFeatureSimpleGA.test_basic
    :layout: interleave

SimpleGADriver Options
----------------------

.. embed-options::
    openmdao.drivers.genetic_algorithm_driver
    SimpleGADriver
    options

You can change the number of generations to run the genetic algorithm by setting the "max_gen" option.

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.TestFeatureSimpleGA.test_option_max_gen
    :layout: interleave

You can change the population size by setting the "pop_size" option. The default value for pop_size is 0,
which means that the driver automatically computes a population size that is 4 times the total number of
bits for all variables encoded.

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.TestFeatureSimpleGA.test_option_pop_size
    :layout: interleave

If you have a model that doesn't contain any distributed components or parallel groups, then the model
evaluations for a new generation can be performed in parallel by turning on the "parallel" option:

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.MPIFeatureTests.test_option_parallel
    :layout: interleave

.. tags:: Driver, Optimizer, Optimization
