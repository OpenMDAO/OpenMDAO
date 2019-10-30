.. _reading_case_data:

*****************************************
Accessing Recorded Data from Case Objects
*****************************************

The :class:`Case<openmdao.recorders.case.Case>` object contains all the information about a specific Case recording whether it was recorded by
a problem, driver, system, or solver. :code:`Case` objects have a number methods for accessing variables and their values.

Example of Getting Variable Data from Case Recording of a Driver
----------------------------------------------------------------

Here are the methods typically used when retrieving data from the recording of a :code:`Driver`.

.. automethod:: openmdao.recorders.case.Case.get_objectives
    :noindex:

.. automethod:: openmdao.recorders.case.Case.get_constraints
    :noindex:

.. automethod:: openmdao.recorders.case.Case.get_design_vars
    :noindex:

.. automethod:: openmdao.recorders.case.Case.get_responses
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

Here are the methods typically used when retrieving data from the recording of a :code:`Problem`.

.. automethod:: openmdao.recorders.case.Case.list_inputs
    :noindex:

.. automethod:: openmdao.recorders.case.Case.list_outputs
    :noindex:

The following example shows how to use these methods.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_list_inputs_and_outputs
    :layout: interleave

The :code:`Case.list_inputs` and :code:`Case.list_outputs` methods have optional arguments that let you filter based on
variable names what gets listed. This is shown in these examples.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_list_inputs_and_outputs_with_includes_excludes
    :layout: interleave

Finally, you can also make use of the variable tagging feature when getting values from cases. This example shows how to do
that.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_list_inputs_and_outputs_with_tags
    :layout: interleave

Getting Variable Data from Case By Specifying Variable Name and Units Desired
-----------------------------------------------------------------------------

You can also get variable values from a :code:`Case` like you would from a :code:`Problem` using dictionary-like access
or, if you want the value in different units, using the :code:`get_val` method.

.. automethod:: openmdao.recorders.case.Case.get_val
    :noindex:

This example shows both methods of getting variable data by name.

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
