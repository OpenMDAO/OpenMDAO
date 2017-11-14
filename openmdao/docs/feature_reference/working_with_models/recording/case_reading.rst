***************************
Working with Recorded Data
***************************

A class, `CaseReader`, is provided to read the data from a case recorder file. It will work for any kind of case
recorder file in OpenMDAO. Currently, OpenMDAO only has a Sqlite case recorder file, but in the future will also have
an HDF5 case recorder file. `CaseReader` should work for either kind of file as it abstracts away the underlying file
format.

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

Assume that a recorder was attached to the `Driver` for the `Problem`. Then, to find out how many cases were recorded:

.. code-block:: console

    print('Number of driver cases recorded =', cr.driver_cases.num_cases )

You can get a list of the case IDs using the `list_cases` method:

.. code-block:: console

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:
        print('Case:', case_key)

Finally, the `get_case` method provides a way to get at individual cases. The argument to this method can either be:

    #. integer - in which case the argument is an index into the cases. Negative numbers can be used as indices just
            as is normally done in Python
    #. string - in which case the argument is one of the case keys

For example, in the common situation where the user wants to see the last case, they can do

.. code-block:: console

    last_case = cr.driver_cases.get_case(-1)
    print('Last value of pz.z =', last_case.desvars['pz.z'])

Or, if the case key is known:

.. code-block:: console

    seventh_slsqp_iteration_case = cr.driver_cases.get_case('rank0:SLSQP|6')
    print('Value of pz.z after 7th iteration of SLSQP =', seventh_slsqp_iteration_case.desvars['pz.z'])