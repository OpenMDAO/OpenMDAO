.. _getting_values:

***********************************
Getting Values with the Case Reader
***********************************

A `CaseReader` class is provided to read the data from a case recorder file. Currently, OpenMDAO only has a
single format of `CaseRecorder`, `SqliteRecorder`.  In the future, we will implement an `HDF5Recorder`, but `CaseReader`
will work for any kind of recorded file, as it abstracts away the underlying file format.

Here is some simple code showing how to use the `CaseReader` class.

.. code-block:: console

    from openmdao.recorders.case_reader import CaseReader

    cr = CaseReader(filename)

Depending on how the cases were recorded and what options were set on the recorder, the case recorder file could contain
any of the following:

    #. Problem metadata
    #. System metadata
    #. Solver metadata
    #. Driver iterations
    #. System iterations
    #. Solver iterations
    #. Driver derivatives

Retrieving Cases
----------------

Assume that a recorder was attached to the `Driver`. You can get a list of the case IDs using the :code:`list_cases` method:

.. code-block:: console

    case_keys = cr.list_cases()
    for case_key in case_keys:
        print('Case:', case_key)

The :code:`get_case` method provides a way to get at individual cases.

.. automethod:: openmdao.recorders.base_case_reader.BaseCaseReader.get_case
    :noindex:

The argument to this method can either be:

    #. String - in which case the argument is one of the case keys.
    #. Integer - in which case the argument is an index into the cases. Negative numbers can be used as indices just
            as is normally done in Python.

For example, in the common situation where you want to see the last case, you can do:

.. code-block:: console

    last_case = cr.get_case(-1)
    print('Last value of z =', last_case.outputs['z'])

Or, if the case key is known:

.. code-block:: console

    seventh_slsqp_iteration_case = cr.driver_cases.get_case('rank0:SLSQP|6')
    print('Value of z after 7th iteration of SLSQP =', seventh_slsqp_iteration_case.outputs['z'])

Note that we can access variables in the case via the either the absolute or promoted names.

.. code-block:: console

    seventh_slsqp_iteration_case = cr.driver_cases.get_case('rank0:SLSQP|6')
    print('Value of pz.z after 7th iteration of SLSQP =', seventh_slsqp_iteration_case.desvars['pz.z'])

Getting Variables and Values
----------------------------
Case objects have a number of attributes and methods to for accessing variables and their values.

.. autoclass:: openmdao.recorders.case.Case
    :members:
    :noindex:

Note that you can use either the promoted or absolute names when accessing the variables.

Example of Using VOI Methods
----------------------------

This example shows how to use the methods to easily see check the variables of interest from the first driver iteration.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_driver_options_with_values
    :layout: interleave


Example of Using List Methods
-----------------------------

This example shows how to use the list methods to view the inputs and outputs for a system case.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_list_inputs_outputs
    :layout: interleave


Loading Cases into Problems
---------------------------

There are some situations where it would be useful to load in a recorded case back into a Problem. The
:code:`Problem.load_case` method is provided for this. The :code:`Problem.load_case` method supports loading in all
forms of cases including systems, drivers, and solver cases.

One possible use case is if you have a long running optimization and, for whatever reason, the run dies before it
completes. It would be great to go back to the last recorded case for the entire model System, load it in to the
Problem, and then do some debugging to determine what went wrong.

Here is an example that shows how you might want to use this method. Notice that this code actually has two separate
runs of the model.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_load_system_case_for_restart
    :layout: interleave

Loading a DataBase into Memory
------------------------------

Every time the `get_case` method is used, the case reader is making a new query to the database 
This doesn't pose a problem when you only intend to access a small subset of the cases or the
database is already small, but can be very slow when you're requesting many cases from a large
recording. To increase efficiency in this scenario you can specify `pre_load=True` when instantiating
the CaseReader, which will load all cases into memory so that subsequent accesses will be fast.

.. code-block:: console

    cr = CaseReader("cases.sql", pre_load=True)
