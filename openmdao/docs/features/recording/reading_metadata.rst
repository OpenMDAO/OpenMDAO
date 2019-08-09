.. _reading_case_metadata:

***************************
Accessing Recorded Metadata
***************************

In addition to the cases themselves, a :ref:`CaseReader<case_reader>` may also record
certain metadata about the model and it's consituent systems and solvers.

Problem Metadata
----------------

By default, a case recorder will save metadata about the model to assist in later visualization
and debugging.  This information is made available via the :code:`problem_metadata` attribute of
a `CaseReader`.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_problem_metadata
    :layout: interleave

System Metadata
----------------

Systems record both scaling factors and options within 'scaling_factors' and 'component_options',
respectively, in :code:`system_metadata`.

The component options include user-defined options that were defined
through the :code:`system.options.declare` method. By default, everything in options is
pickled and recorded. If there are options that cannot be pickled or you simply do not wish
to record, they can be excluded using the 'options_excludes' recording option on the system.

By setting the :code:`record_model_metadata` option on `Driver`, you can record the metadata
for every system in the model.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_system_metadata
    :layout: interleave

.. note::
    Each system object must have a recorder explicitly attached in order for its metadata and options to be recorded.

   
Solver Metadata
---------------

Solvers record the solver options as metadata. Note that, because more than
one solver's metadata may be recorded, each solver's metadata must be accessed through
its absolute name within :code:`solver_metadata`, as shown in the example below.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_metadata
    :layout: interleave
