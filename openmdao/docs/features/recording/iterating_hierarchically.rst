***********************************
Iterating Over Case Hierarchies
***********************************

The Case Reader offers a number of ways to iterate over cases. Perhaps
the simplest is using the :code:`list_cases` method with :code:`get_case`
to iterate over all cases of a certain type.

.. code-block:: console

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:
        case = cr.driver_cases.get_case(case_key)
        ...

The Case Reader also has a :code:`get_cases` method, which provides a way to iterate
over Driver and Solver cases in order with a choice of iterating hierarchically or flat
(using the :code:`recurse` parameter).

.. automethod:: openmdao.recorders.sqlite_reader.SqliteCaseReader.get_cases
    :noindex:

For example, if you wanted to iterate over all Driver and Solver cases that were recorded
you could use:

.. code-block:: console

    for case in cr.get_cases(recursive=True):
        timestamp = case.timestamp
        ...

If you wanted to iterate over all solver cases that are direct descendents of the first driver case you could use:

.. code-block:: console

    for case in cr.get_cases(parent='rank0:SLSQP|0', recursive=False):
        timestamp = case.timestamp
        ...

.. note::
    This generator can return both `Driver` and `Solver` cases, which store similar information (`inputs`, `outputs`, `residuals`, etc.)
    but aren't identical. In particular, `Solver` cases additionally store `abs_err` and `rel_err`.