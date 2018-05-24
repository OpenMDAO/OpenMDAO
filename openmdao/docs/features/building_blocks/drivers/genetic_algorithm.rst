
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

Running a GA in Parallel
------------------------

If you have a model that doesn't contain any distributed components or parallel groups, then the model
evaluations for a new generation can be performed in parallel by turning on the "parallel" option:

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.MPIFeatureTests.test_option_parallel
    :layout: interleave

Running a GA on a Parallel Model in Parallel
--------------------------------------------

If you have a model that does contain distributed components or parallel groups, you can also use
`TestFeatureSimpleGA` to optimize it. If you have enough processors, you can also simultaneously
evaluate multiple points in your population by turning on the "paralllel" option and setting the
"procs_per_model" to the number of processors that your model requires. Take care that you submit
your parallel run with enough processors such that the number of processors the model requires
divide evenly into it, as in this example, where the model requires 2 and we give it 4.

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.MPIFeatureTests4.test_option_procs_per_model
    :layout: interleave

.. tags:: Driver, Optimizer, Optimization
