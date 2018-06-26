***********************************
Getting Values with the Case Reader
***********************************

A `CaseReader` class is provided to read the data from a case recorder file. Currently, OpenMDAO only has a
single format of `CaseRecorder`, `SqliteRecorder`.  In the future, we will implement an `HDF5Recorder`, but `CaseReader`
will work for any kind of recorded file, as it abstracts away the underlying file format.

Here is some simple code showing how to use the `CaseReader` class.

.. code-block:: console

    from openmdao.recorders.case_reader import CaseReader

    cr = CaseReader(case_recorder_filename)

Depending on how the cases were recorded and what options were set on the recorder, the case recorder file could contain
any of the following:

    #. Driver metadata
    #. System metadata
    #. Solver metadata
    #. Driver iterations
    #. System iterations
    #. Solver iterations

Retrieving Cases
----------------

Assume that a recorder was attached to the `Driver` for the `Problem`. To find out how many cases were recorded:

.. code-block:: console

    print('Number of driver cases recorded =', cr.driver_cases.num_cases )

You can get a list of the case IDs using the :code:`list_cases` method:

.. code-block:: console

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:
        print('Case:', case_key)

The :code:`get_case` method provides a way to get at individual cases.

.. automethod:: openmdao.recorders.sqlite_reader.DriverCases.get_case
    :noindex:

The argument to this method can either be:

    #. Integer - in which case the argument is an index into the cases. Negative numbers can be used as indices just
            as is normally done in Python.
    #. String - in which case the argument is one of the case keys.

For example, in the common situation where the user wants to see the last case, they can do:

.. code-block:: console

    last_case = cr.driver_cases.get_case(-1)
    print('Last value of pz.z =', last_case.desvars['z'])

Or, if the case key is known:

.. code-block:: console

    seventh_slsqp_iteration_case = cr.driver_cases.get_case('rank0:SLSQP|6')
    print('Value of pz.z after 7th iteration of SLSQP =', seventh_slsqp_iteration_case.desvars['z'])

Note that we access variables in the case reader through the promoted names instead of the absolute variable name.
If we had not promoted `pz.z`, we would use:

.. code-block:: console

    seventh_slsqp_iteration_case = cr.driver_cases.get_case('rank0:SLSQP|6')
    print('Value of pz.z after 7th iteration of SLSQP =', seventh_slsqp_iteration_case.desvars['pz.z'])

Getting Variables and Values
----------------------------
Both the CaseReader and cases themselves have a number of methods to retrieve types of variables. On Case objects there are the methods :code:`get_desvars()`, :code:`get_objectives()`, :code:`get_constraints()`,
and :code:`get_responses()` which, as their names imply, will return the corresponding set of variables and their values on that case.

Here's an example that shows how to use these methods to see what variables were recorded on the first driver iteration and get their values.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_driver_options_with_values
    :layout: interleave

Additionally, just like :ref:`listing variables <listing-variables>` on System objects, there is a :code:`list_inputs` method and a :code:`list_outputs` method on the CaseReader.

.. automethod:: openmdao.recorders.sqlite_reader.SqliteCaseReader.list_inputs
    :noindex:

.. automethod:: openmdao.recorders.sqlite_reader.SqliteCaseReader.list_outputs
    :noindex:

These methods default to using System cases if no specific case is supplied. If the user does supply a Case then the output will only reflect the variables recorded within that case. For example, if we wanted to get the inputs and outputs recorded in the last driver case we would use:

.. code-block:: console

    last_driver_case = cr.driver_cases.get_case(-1)
    inputs = cr.list_inputs(last_driver_case)
    outputs = cr.list_outputs(last_driver_case)

By default, both methods will give all inputs or outputs recorded in system iterations and, if the `values` parameter is set to True, the last recorded value of each variable. Grabbing all recorded inputs and outputs is as simple as:

.. code-block:: console

    all_outputs = cr.list_outputs()
    all_inputs = cr.list_inputs()

Additionally, for quick access to values recorded there are :code:`inputs` and :code:`outputs` dictionaries on every Case and :code:`residuals` on System and Solver cases. If you wanted to quickly grab the value and residual of output `x`
on the last solver iteration you would use:

.. code-block:: console

    last_solver_case = cr.solver_cases.get_case(-1)
    x_val = last_solver_case.outputs['x']
    x_residual = last_solver_case.residuals['x']

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

Every time the `get_case` method is used, the case reader is making a new query
to the DB (with the exception of recurring requests, which are cached). This doesn't
pose a problem when you only intend to access a small subset of the cases or the DB is
already small, but can be very slow when you're requesting many cases from a large
recording. To increase efficiency in this scenario you should use the CaseReader's
:code:`load_cases` method, which loads all driver, solver, and system cases into memory
with minimal queries.

To use this method, simply create the CaseReader, call the `load_cases` method, and use the
reader as you normally would.

.. code-block:: console

    cr = CaseReader('cases.sql')
    cr.load_cases()
    ...
