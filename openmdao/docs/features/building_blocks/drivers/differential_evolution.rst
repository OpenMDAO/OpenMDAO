
***************************
DifferentialEvolutionDriver
***************************

.. note::
    DifferentialEvolutionDriver is based on SimpleGADriver and supports most of the same options and capabilities.

This `differential evolution <http://en.wikipedia.org/wiki/Differential_evolution>`_ variant of a genetic algorithm
optimizer supports only continuous variables.
The DifferentialEvolutionDriver supports both constrained and unconstrained optimization.

The DifferentialEvolutionDriver has advantages and disadvantages when compared to the SimpleGADriver:

* Pros

  * DifferentialEvolutionDriver is typically about 3 times faster than SimpleGADriver

  * DifferentialEvolutionDriver is usually more accurate than SimpleGADriver because it does not limit the number of bits available to represent inputs

  * DifferentialEvolutionDriver does not require the user to manually specify a number of representation bits

* Cons

  * DifferentialEvolutionDriver only supports continuous input variables; SimpleGADriver also supports discrete

  * DifferentialEvolutionDriver does not support SimpleGADriver's "compute_pareto" option for multi-objective optimization

Genetic algorithms do not use gradient information to find optimal solutions. This makes them ideal
for problems that do not have gradients or problems with many local minima where gradient information
is not helpful in finding the global minimum. A well known example of this is finding the global minimum of
of the `Rastrigin function <http://en.wikipedia.org/wiki/Rastrigin_function>`_:

  .. figure:: images/rastrigin2d.png
     :align: center
     :width: 100%
     :alt: 2 Dimensional Rastigin Function

The example below shows an OpenMDAO solution of a higher order Rastrigin function.

.. embed-code::
    openmdao.drivers.tests.test_differential_evolution_driver.TestDifferentialEvolution.test_rastrigin
    :layout: interleave

DifferentialEvolutionDriver Options
-----------------------------------

.. embed-options::
    openmdao.drivers.differential_evolution_driver
    DifferentialEvolutionDriver
    options

DifferentialEvolutionDriver Constructor
---------------------------------------

The call signature for the `DifferentialEvolutionDriver` constructor is:

.. automethod:: openmdao.drivers.differential_evolution_driver.DifferentialEvolutionDriver.__init__
    :noindex:

Using DifferentialEvolutionDriver
---------------------------------

You can change the number of generations to run the genetic algorithm by setting the "max_gen" option.

.. embed-code::
    openmdao.drivers.tests.test_differential_evolution_driver.TestFeatureDifferentialEvolution.test_option_max_gen
    :layout: interleave

You can change the population size by setting the "pop_size" option. The default value for pop_size is 0,
which means that the driver automatically computes a population size that is 20 times the total number of
input variables.

.. embed-code::
    openmdao.drivers.tests.test_differential_evolution_driver.TestFeatureDifferentialEvolution.test_option_pop_size
    :layout: interleave
