.. _reading_case_data:

***********************
Accessing Recorded Data
***********************

A :ref:`CaseReader<iterating_case_data>` provides access to the data for a particular case via a
:class:`Case<openmdao.recorders.case.Case>` object.

Getting Variable Data from a Case
---------------------------------

`Case` objects have a number of attributes and methods to for accessing variables and their values.

The following example shows how to use these methods to easily check the variables of interest 
from the first driver iteration.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_driver_options_with_values
    :layout: interleave

.. note::
    Note that you can use either the promoted or absolute names when accessing variables.


Getting Derivative Data from a Case
-----------------------------------

A driver has the ability to record derivatives but it is not enabled by default. If you do enable 
this option, the recorded cases will have a value for the :code:`jacobian`.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_reading_derivatives
    :layout: interleave
