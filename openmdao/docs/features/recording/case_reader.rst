.. _case_reader:

******************
Case Reader Object
******************

The :code:`CaseReader` object is provided to read case recordings no matter what case recorder was used.
Currently, OpenMDAO only supports the :code:`SqliteCaseReader` case recorder. Therefore, all the examples will
make use of this recorder. OpenMDAO will support other case recorders in the future.

.. automethod:: openmdao.recorders.sqlite_reader.SqliteCaseReader.__init__
    :noindex:

Determining What Sources and Variables Were Recorded
----------------------------------------------------

The :code:`CaseReader` object provides methods to determine which objects in the original problem were responsible
for recording cases and what variables they recorded.

The :code:`list_sources` method provides a
list of the names of objects that are the sources of the recorded data in the file.

.. automethod:: openmdao.recorders.base_case_reader.BaseCaseReader.list_sources
    :noindex:

The complementary :code:`list_source_vars` method will provide a dictionary of the input and output variables recorded
by each of those sources.

.. automethod:: openmdao.recorders.base_case_reader.BaseCaseReader.list_source_vars
    :noindex:

Here is an example of their usage.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_list_sources
    :layout: interleave


Case Names
----------

:code:`CaseReader` object contain a collection of :code:`Case` objects, one for each individual recording of an object
which can be anyone of these objects: driver, problem, component, or solver.

:code:`Case` objects are uniquely identified in a case recorder file by their case names. A Case name is a string.
As an example, here is a case name:

    'rank0:ScipyOptimize_SLSQP|1|root._solve_nonlinear|1'

The first part of the case name indicates which rank or process that the case was recorded from. The remainder of the
case name shows the hierarchical path to the object that was recorded along with the iteration counts for each object
along the path. It follows a pattern of repeated pairs of

    - object name ( problem, driver, system, or solver )
    - iteration count

These are separated by the :code:`|` character.

So in the given example, the case is:

    - from rank 0
    - the first iteration of the driver, `ScipyOptimize_SLSQP`
    - the first execution of the `root` system which is the top-level model

Getting Names of the Cases
--------------------------

The :code:`list_cases` method returns the names of the cases in the order in which
the cases were executed. You can optionally request cases only from a specific :code:`source`.

.. automethod:: openmdao.recorders.base_case_reader.BaseCaseReader.list_cases
    :noindex:

.. _list_cases_args:

There are two optional arguments to the :code:`list_cases` method that effect what is returned.

    - :code:`recurse`: causes the returned value to include child cases.

    - :code:`flat`: works in conjunction with the :code:`recurse` argument to determine if the returned
      results are in the form of a list or nested dict. If recurse=True, flat=False, and there are child cases, then
      returned value is a nested ordered dict. Otherwise, it is a list.


Getting Access to Cases Using get_cases Method
----------------------------------------------

Getting information from the cases is a two-step process. First, you need to get access to the Case object and then
you can call a variety of methods on the Case object to get values from it. The second step is described on the
:ref:`Accessing Recorded Data<reading_case_data>` page.

There are two methods used to get access to :code:`Cases`:

    - :code:`get_cases`
    - :code:`get_case`


Getting Access to Cases Using get_cases Method
----------------------------------------------

The :code:`get_cases` method provides a quick and easy way to iterate over all the cases.

.. automethod:: openmdao.recorders.base_case_reader.BaseCaseReader.get_cases
    :noindex:

The method :code:`get_cases` is similar to the :code:`list_cases` method in that it has the two optional arguments
:code:`recurse` and :code:`flat` to control what is returned and the data structure returned. See
:ref:`explanation of the list_cases args<list_cases_args>`.

Here is an example of its usage.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_get_cases
    :layout: code, output

Getting Access to the Case Values Using get_case Method
-------------------------------------------------------

The :code:`get_case` method returns a :code:`Case` object given a case name.

.. automethod:: openmdao.recorders.base_case_reader.BaseCaseReader.get_case
    :noindex:

If after getting the list of case names using the :code:`list_cases` method, you determine the case you want information
from, you can use the :code:`get_case` method to get that case.

This code snippet shows how to get the first case.

.. code::

    cr = om.CaseReader('cases.sql')
    case_names = cr.list_cases()
    case = cr.get_case(case_names[0])

You could also loop over all the case names and get access to the cases as shown in this example.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_list_cases
    :layout: code, output

Processing a Nested Dictionary of Its Child Cases
-------------------------------------------------
The following example demonstrates selecting a case from a case list and processing a nested
dictionary of its child cases.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_get_cases_nested
    :layout: code, output

