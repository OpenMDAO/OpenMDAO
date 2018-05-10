**************************
Recording and Reading Data
**************************

One of the most useful features in OpenMDAO is :ref:`Case Recording <case_recording>`. With case recording you can
record variables and their values over the course of an optimization and store the
information in a local database file, then read the values using a Case Reader.
This can be used for debugging, plotting/visualization, and even starting a new optimization
where an old one left off.

To record, all you have to do is create a case recorder (currently OpenMDAO only supports a SQLite case recorder),
then attach it to any drivers, systems, or solvers whose data you want to store.
In the example below, we create the recorder with :code:`recorder = SqliteRecorder(self.filename)`,
where the filename is a string, then attached it to the driver with :code:`p.driver.add_recorder(recorder)`

Once we run the optimization we can read the data from the file using a case reader.
As can be seen below, we do this by creating a `CaseReader` object and pass in the filename
of our generated SQLite database: :code:`cr = CaseReader(self.filename)`. With the case reader
ready we then grab the data recorded on each iteration using :code:`cr.driver_cases.get_case`
and access the `inputs`, `outputs`, and/or `residuals`.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestSqliteCaseReader.test_circuit_with_recorder
    :layout: interleave
