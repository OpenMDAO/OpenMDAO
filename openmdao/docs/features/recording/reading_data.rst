.. _reading_case_data:

***********************
Accessing Recorded Data
***********************

A :ref:`CaseReader<iterating_case_data>` provides access to the data for a particular case via a
:class:`Case<openmdao.recorders.case.Case>` object.

`Case` objects have a number of attributes and methods for accessing variables and their values.

Example of Getting Variable Data from Case Recording of a Driver
----------------------------------------------------------------

Here are the methods typically used when retrieving data from the recording of a `Driver`.

.. automethod:: openmdao.recorders.case.Case.get_objectives
    :noindex:

.. automethod:: openmdao.recorders.case.Case.get_constraints
    :noindex:

.. automethod:: openmdao.recorders.case.Case.get_design_vars
    :noindex:

The following example shows how to use these methods to easily check the variables of interest
from the first driver iteration.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_driver_options_with_values
    :layout: interleave

.. note::
    Note that you can use either the promoted or absolute names when accessing variables.

Getting Variable Data from Case Recording of a Problem
------------------------------------------------------

Here are the methods typically used when retrieving data from the recording of a `Problem`.

.. automethod:: openmdao.recorders.case.Case.list_inputs
    :noindex:

.. automethod:: openmdao.recorders.case.Case.list_outputs
    :noindex:

The following example shows how to use these methods.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_list_inputs_and_outputs
    :layout: interleave

You can also get variable values from a Case like you would from a Problem using dictionary-like access
or, if you want the value in different units, using the :code:`get_val` method:

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_get_val
    :layout: interleave
    
Getting Derivative Data from a Case
-----------------------------------

A driver has the ability to record derivatives but it is not enabled by default. If you do enable
this option, the recorded cases will have a value for the :code:`jacobian`.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_reading_derivatives
    :layout: interleave
