
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

The SimpleGADriver supports both constrained and unconstrained optimization.

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

Constrained Optimization
------------------------

The SimpleGADriver supports both constrained and unconstrained optimization. If you have constraints,
the constraints are added to the objective after they have been weighted using a user-tunable
penalty mutiplier and exponent.

        All constraints are converted to the form of :math:`g(x)_i \leq 0` for
        inequality constraints and :math:`h(x)_i = 0` for equality constraints.
        The constraint vector for inequality constraints is the following:

        .. math::

           g = [g_1, g_2  \dots g_N], g_i \in R^{N_{g_i}}
           h = [h_1, h_2  \dots h_N], h_i \in R^{N_{h_i}}

        The number of all constraints:

        .. math::

           N_g = \sum_{i=1}^N N_{g_i},  N_h = \sum_{i=1}^N N_{h_i}

        The fitness function is constructed with the penalty parameter :math:`p`
        and the exponent :math:`\kappa`:

        .. math::

           \Phi(x) = f(x) + p \cdot \sum_{k=1}^{N^g}(\delta_k \cdot g_k^{\kappa})
           + p \cdot \sum_{k=1}^{N^h}|h_k|^{\kappa}

        where :math:`\delta_k = 0` if :math:`g_k` is satisfied, 1 otherwise

The following example shows how to set the penalty parameter :math:`p` and the exponent :math:`\kappa`:

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.TestFeatureSimpleGA.test_constrained_with_penalty
    :layout: code, output


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
`SimpleGADriver` to optimize it. If you have enough processors, you can also simultaneously
evaluate multiple points in your population by turning on the "parallel" option and setting the
"procs_per_model" to the number of processors that your model requires. Take care that you submit
your parallel run with enough processors such that the number of processors the model requires
divides evenly into it, as in this example, where the model requires 2 and we give it 4.

.. embed-code::
    openmdao.drivers.tests.test_genetic_algorithm_driver.MPIFeatureTests4.test_option_procs_per_model
    :layout: interleave

.. tags:: Driver, Optimizer, Optimization
