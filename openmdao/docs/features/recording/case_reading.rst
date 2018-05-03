**************************
Working with Recorded Data
**************************

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

Assume that a recorder was attached to the `Driver` for the `Problem`. To find out how many cases were recorded:

.. code-block:: console

    print('Number of driver cases recorded =', cr.driver_cases.num_cases )

You can get a list of the case IDs using the `list_cases` method:

.. code-block:: console

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:
        print('Case:', case_key)

The `get_case` method provides a way to get at individual cases. The argument to this method can either be:

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

If a user would like to access the user-defined options on a given system or the scaling factors, the CaseReader also has a `system_metadata` dictionary. The case recorder does need to be explicitly added to a system in order for its metadata to be recorded. System metadata stored in `system_metadata` is accessed by system name where each system has two keys: `component_metadata` for accessing the user-defined options and `scaling_factors`. Accessing this data for `pz` would look like:

.. code-block:: console

    pz.options.declare('test data', True)
    ...
    pz_metadata = cr.system_metadata['pz']['component_metadata']
    test_data_value = pz_metadata['test data']

    pz_scaling = cr.system_metadata['pz']['scaling_factors']



Finally, if a user would like to access variable metadata there is a `output2meta` dictionary and a `input2meta` dictionary on the CaseReader. For example, if the user wanted the units of the `pz.z` variable they would use:

.. code-block:: console

    z_units = cr.output2meta['z']['units']

*Iterating Over Cases*
~~~~~~~~~~~~~~~~~~~~~~

The :code:`get_cases` method provides a way to iterate over Driver and Solver cases in order.

.. automethod:: openmdao.recorders.sqlite_reader.SqliteCaseReader.get_cases
    :noindex:

For example, if the user wanted to iterate over all Driver and Solver cases they would use:

.. code-block:: console

    for case in cr.get_cases(recursive=True):
        timestamp = case.timestamp
        ...

If the user wanted to iterate over all solver cases that are descendents of the first driver case they could use:

.. code-block:: console

    for case in cr.get_cases(parent='rank0:SLSQP|0', recursive=True):
        timestamp = case.timestamp
        ...

Note that this generator can return both Driver and Solver cases, which have different attributes.

*Listing Variables*
~~~~~~~~~~~~~~~~~~~
Both the CaseReader and cases themselves have a number of methods to retrieve types of variables. On Case objects there are the methods :code:`get_desvars()`, :code:`get_objectives()`, :code:`get_constraints()`, and :code:`get_responses()` which, as their names imply, will return the corresponding set of variables and their values on that case. In the situation where the user may want to print the design variables on the first driver case they could use:

.. code-block:: console

    print("-----------desvars-----------")
    first_case = cr.driver_cases.get_case(0)
    desvars = first_case.get_desvars()
    for des in desvars:
        print(des + ': ' + str(desvars[des]))

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
    all_inputs cr.list_outputs()
