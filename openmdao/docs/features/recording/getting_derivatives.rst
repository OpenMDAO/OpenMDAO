****************************************
Getting Derivatives with the Case Reader
****************************************

If you are running with a driver that computes derivatives, you can also record and read them through the `CaseReader`.
The derivatives are also stored in cases, though the structure of a derivatives case is simpler than for iteration
cases. The `CaseReader` contains a "driver_derivative_cases" object which contains a case for each time the driver
computed derivatives. The case keys are iterations coordinates. Derivatives are recorded using the same iteration
coordinate at which the model was executed and linearized. Each case has the method "get_derivatives" to provide a
dictionary with all derivatives for that specific case.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestSqliteCaseReader.test_feature_reading_derivatives
    :layout: interleave