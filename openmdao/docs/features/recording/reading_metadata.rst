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
These values are accessible using code such as this, where :code:`cr` is a case reader object.

    - :code:`cr.list_model_options()`

This function creates a dictionary of the latest model options available that were set in the model.


.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_recording_system_options
    :layout: interleave


Solver Metadata
---------------

Solvers record the solver options as metadata. Note that, because more than
one solver's metadata may be recorded, each solver's metadata must be accessed through
its absolute name within :code:`solver_metadata`, as shown in the example below.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_metadata
    :layout: interleave
