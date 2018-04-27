.. _doe_driver:

*********
DOEDriver
*********

`DOEDriver` facilitates performing a design of experiments (DOE) with your OpenMDAO model.
It will run your model multiple times with different values for the design variables
depending on the selected input generator. A number of generators are available, each with
its own parameters that can be specified when it is instantiated:

* :class:`UniformGenerator<openmdao.drivers.doe_generators.UniformGenerator>`
* :class:`FullFactorialGenerator<openmdao.drivers.doe_generators.FullFactorialGenerator>`
* :class:`PlackettBurmanGenerator<openmdao.drivers.doe_generators.PlackettBurmanGenerator>`
* :class:`BoxBehnkenGenerator<openmdao.drivers.doe_generators.BoxBehnkenGenerator>`
* :class:`LatinHypercubeGenerator<openmdao.drivers.doe_generators.LatinHypercubeGenerator>`

.. note::
    All generators except for `UniformGenerator` are provided via the `pyDOE2`_ package,
    which is an updated version of `pyDOE`_.  See the original `pyDOE`_ page for
    information on those algorithms.

The generator instance is supplied as an argument to the `DOEDriver`.


Simple Example
--------------
UniformGenerator implements the simplest method and will generate a requested number of
samples randomly selected from a uniform distribution across the valid range for each
design variable. This example demonstrates its use with a model built on the
:ref:`Paraboloid<tutorial_paraboloid_analysis>` Component.
An :ref:`SqliteRecorder<basic_recording>` is used to capture the cases that were generated.
We can see that that the model was evaluated at random values of :code:`x` and :code:`y`
between -10 and 10, per the lower and upper bounds of those design variables.

.. embed-code::
    openmdao.drivers.tests.test_doe_driver.TestDOEDriverFeature.test_uniform
    :layout: interleave


DOEDriver Options
-----------------

.. embed-options::
    openmdao.drivers.doe_driver
    DOEDriver
    options


.. _doe_driver_parallel:

Running a DOE Driver in Parallel
--------------------------------

In a parallel processing environment, it is possible for `DOEDriver` to run
cases concurrently. This is done by specifying the `parallel` option as shown
in the following example.

Here we are using the `FullFactorialGenerator` with 3 levels to generate inputs
for our `Paraboloid` model. With two inputs, :math:`3^2=9` cases have been
generated. Since we are running on two processors, those cases have been split
with 5 cases run on the first processor and 4 cases on the second.

Note that, when running in parallel, the `SqliteRecorder` will generate a separate
case file for each processor on which a case is recorded. A message will be displayed
indicating the name of each file, as seen in the example.

.. embed-code::
    openmdao.drivers.tests.test_doe_driver.TestParallelDOEFeature.test_full_factorial
    :layout: interleave

.. _pyDOE: https://pythonhosted.org/pyDOE
.. _pyDOE2: https://pypi.org/project/pyDOE2

.. tags:: Driver, DOE
