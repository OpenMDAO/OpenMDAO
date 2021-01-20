.. _reading_case_metadata:

***************************
Accessing Recorded Metadata
***************************

In addition to the cases themselves, a :ref:`CaseReader<case_reader>` may also record
certain metadata about the model and its constituent systems and solvers.

Problem Metadata
----------------

By default, a case recorder will save metadata about the model to assist in later visualization
and debugging.  This information is made available via the :code:`problem_metadata` attribute of
a `CaseReader`.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_problem_metadata
    :layout: interleave

System Options
--------------

All case recorders record the component options and scaling factors for all systems in the model.
These values are accessible using the :code:`list_model_options` function of a case reader object.
This function displays and returns a dictionary of the option values for each system in the model.
If the model has been run multiple times, you can specify the run for which to get/display options.


.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_system_options
    :layout: interleave


Solver Options
--------------

All case recorders record the solver options for all solvers in the model.
These values are accessible using the :code:`list_solver_options` function of a case reader object.
This function displays and returns a dictionary of the option values for each solver in the model.
If the model has been run multiple times, you can specify the run for which to get/display options.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_options
    :layout: interleave
