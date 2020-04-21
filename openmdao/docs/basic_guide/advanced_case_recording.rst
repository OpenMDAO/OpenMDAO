.. _advanced_case_recording:

***************************
Advanced Recording Example
***************************

There are customizations a user can make to the recorder to control what to record and not to record. Below
we demonstrate how a user can change the True/False values of various recording options to specify exactly
what should be recorded on a driver/problem/solver/component.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_advanced_example
    :layout: interleave

Isolating Source and Case
--------------------------

OpenMDAO's case reader allows you to isolate a specific case within a source. Using the example from
above, we'll show how to isolate a case from driver. In the example you can replace driver
in `cases = cr.get_cases('driver')` with problem to access a different source.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_isolate_case
    :layout: interleave
