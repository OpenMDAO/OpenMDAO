***********************************
Iterating Over Case Hierarchies
***********************************

The Case Reader offers a number of ways to iterate over cases. Perhaps
the simplest is using the :code:`list_cases` method with :code:`get_case`
to iterate over all cases of a certain type.

.. code-block:: console

    case_keys = cr.list_cases()
    for case_key in case_keys:
        case = cr.get_case(case_key)
        ...

The Case Reader also has a :code:`get_cases` method, which provides a list of
Driver, System and Solver cases in order, or a hiearchical dictionary structure
of those cases.

.. automethod:: openmdao.recorders.sqlite_reader.SqliteCaseReader.get_cases
    :noindex:

For example, if you wanted a list of all cases that were recorded during a run you would use:

.. code-block:: console

    for case in cr.get_cases(recurse=True):
        source = case.source
        outputs = case.outputs
        ...

If you wanted to iterate over all cases that are descendents of the first driver case you could use:

.. code-block:: console

    for case in cr.get_cases('rank0:SLSQP|0'):
        source = case.source
        outputs = case.outputs
        ...
