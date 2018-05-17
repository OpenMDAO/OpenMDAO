**************************
Recording and Reading Data
**************************

One of the most useful features in OpenMDAO is :ref:`Case Recording <case_recording>`. With case recording you can
record variables and their values over the course of an optimization and store the
information in a local database file, then read the values using a Case Reader.
This can be used for debugging, plotting/visualization, and even starting a new optimization
where an old one left off.

In the example below we create a simple circuit model, attach a recorder to the model's driver,
run the optimization, then use a case reader to examine recorded values.

The Run Script
**************

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestSqliteCaseReader.test_circuit_with_recorder
    :layout: interleave

Creating and Attaching a Recorder
---------------------------------

To record, all you have to do is create a case recorder (currently OpenMDAO only supports a SQLite case recorder),
then attach it to any drivers, systems, or solvers whose data you want to store. This
recorder creates a local SQLite database file when the optimization is run.

.. code::

    recorder = SqliteRecorder(case_filename)
    p.driver.add_recorder(recorder)

Creating a Case Reader and Reading Data
-------------------------------------

Once we run the optimization we can read the data from the file using a case reader.
By attaching the recorder to our driver we've recorded the results of each driver iteration
in driver `cases`. To access these cases we use :code:`get_case` on the CaseReader's
:code:`driver_cases`, passing in 0 to indicate that we want the results of the first driver
iteration.

.. code::

    cr = CaseReader(case_filename)
    first_driver_case = cr.driver_cases.get_case(0)

We inspect the values of inputs and outputs by accessing the DriverCase's :code:`inputs` and :code:`outputs`,
respectively.

.. code::

    V_in = first_driver_case.inputs['circuit.R1.V_in']
    I = first_driver_case.outputs['circuit.R1.I']
