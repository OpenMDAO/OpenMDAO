.. _doe_driver:

*********
DOEDriver
*********

:class:`DOEDriver<openmdao.drivers.doe_generators.DOEDriver>` facilitates performing a
design of experiments (DOE) with your OpenMDAO model.
It will run your model multiple times with different values for the design variables
depending on the selected input generator. A number of generators are available, each with
its own parameters that can be specified when it is instantiated:

* :class:`UniformGenerator<openmdao.drivers.doe_generators.UniformGenerator>`
* :class:`FullFactorialGenerator<openmdao.drivers.doe_generators.FullFactorialGenerator>`
* :class:`PlackettBurmanGenerator<openmdao.drivers.doe_generators.PlackettBurmanGenerator>`
* :class:`BoxBehnkenGenerator<openmdao.drivers.doe_generators.BoxBehnkenGenerator>`
* :class:`LatinHypercubeGenerator<openmdao.drivers.doe_generators.LatinHypercubeGenerator>`
* :class:`CSVGenerator<openmdao.drivers.doe_generators.CSVGenerator>`
* :class:`ListGenerator<openmdao.drivers.doe_generators.ListGenerator>`

.. note::
    `FullFactorialGenerator`, `PlackettBurmanGenerator`, `BoxBehnkenGenerator` and
    `LatinHypercubeGenerator` are provided via the `pyDOE2`_ package, which is an
    updated version of `pyDOE`_.  See the original `pyDOE`_ page for information on
    those algorithms.

The generator instance may be supplied as an argument to the `DOEDriver` or as an option.

DOEDriver Options
-----------------

.. embed-options::
    openmdao.drivers.doe_driver
    DOEDriver
    options

Simple Example
--------------
`UniformGenerator` implements the simplest method and will generate a requested number of
samples randomly selected from a uniform distribution across the valid range for each
design variable. This example demonstrates its use with a model built on the
:ref:`Paraboloid<tutorial_paraboloid_analysis>` Component.
An :ref:`SqliteRecorder<saving_data>` is used to capture the cases that were generated.
We can see that that the model was evaluated at random values of :code:`x` and :code:`y`
between -10 and 10, per the lower and upper bounds of those design variables.

.. embed-code::
    openmdao.drivers.tests.test_doe_driver.TestDOEDriverFeature.test_uniform
    :layout: interleave


.. _doe_driver_parallel:

Running a DOE in Parallel
-------------------------

In a parallel processing environment, it is possible for `DOEDriver` to run
cases concurrently. This is done by setting the `run_parallel` option to True as shown
in the following example.

Here we are using the `FullFactorialGenerator` with 3 levels to generate inputs
for our `Paraboloid` model. With two inputs, :math:`3^2=9` cases have been
generated. In this case we are running on two processors and have specified
:code:`options['run_parallel']=True` to run cases on all available processors.
The cases have therefore been split with 5 cases run on the first processor
and 4 cases on the second.

Note that, when running in parallel, the `SqliteRecorder` will generate a separate
case file for each processor on which cases are recorded. The case files will have a
suffix indicating the recording rank and a message will be displayed indicating the
file name, as seen in the example.

.. embed-code::
    openmdao.drivers.tests.test_doe_driver.TestParallelDOEFeature.test_full_factorial
    :layout: interleave


Running a DOE in Parallel with a Parallel Model
-----------------------------------------------

If the model that is being subjected to the DOE is also parallel, then the total
number of processors should reflect the model size as well as the desired concurrency.

To illustrate this, we will demonstrate performing a DOE on a model based on the
:ref:`ParallelGroup<feature_parallel_group>` example:

.. embed-code::
    openmdao.test_suite.groups.parallel_groups.FanInGrouped
    :layout: code

In this case, the model itself requires two processors, so in order to run cases
concurrently we need to allocate at least four processors in total. We can allocate
as many processors as we have available, however the number of processors must be a multiple
of the number of processors per model, which is 2 here. Regardless of how many processors
we allocate, we need to tell the `DOEDriver` that the model needs 2 processors, which
is done by specifying :code:`options['procs_per_model']=2`. From this, the driver
figures out how many models it can run in parallel, which in this case is also 2.

The `SqliteRecorder` will record cases on the first two processors, which serve as
the "root" processors for the parallel cases.

.. embed-code::
    openmdao.drivers.tests.test_doe_driver.TestParallelDOEFeature2.test_fan_in_grouped
    :layout: code, output


Using Prepared Cases
--------------------
If you have a previously generated set of cases that you want to run using `DOEDriver`,
there are a couple of ways to do that. The first is to provide those inputs via an
external file in the CSV (comma separated values) format.  The file should be organized
with one column per design variable, with the first row containing the names of the design
variables. The following example demonstrates how to use such a file to run a DOE using
the `CSVGenerator`:

.. embed-code::
    openmdao.drivers.tests.test_doe_driver.TestDOEDriverFeature.test_csv
    :layout: interleave

The second method is to provide the data directly as a list of cases, where each case is a
collection of name/value pairs for the design variables. You might use this method if you
want to generate the cases programmatically via another algorithm or if the data is
available in some format other than a CSV file and you can reformat it into this simple
list structure. The `DOEGenerator` you would use in this case is the `ListGenerator`,
but if you pass a list directly to the `DOEDriver` it will construct the `ListGenerator`
for you. In the following example, a set of cases has been pre-generated and saved in JSON
(JavaScript Object Notation) format. The data is decoded and provided to the `DOEDriver`
as a list:

.. embed-code::
    openmdao.drivers.tests.test_doe_driver.TestDOEDriverFeature.test_list
    :layout: interleave


.. warning::
    When using pre-generated cases via `CSVGenerator` or `ListGenerator`, there is no
    enforcement of the declared bounds on a design variable as with the algorithmic
    generators.


.. _pyDOE: https://pythonhosted.org/pyDOE
.. _pyDOE2: https://pypi.org/project/pyDOE2

.. tags:: Driver, DOE
