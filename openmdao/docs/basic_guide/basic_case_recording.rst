.. _basic_case_recording:

************************
Basic Recording Example
************************

Recording Terminology
---------------------

| **Case**: A Case stores a snapshot of all the variable values, metadata, and options of a model, or a sub-set of a model, at a particular point in time
| **Case Recorder**: An OpenMDAO module used to store a snapshot of a model before, during, or after execution in an SQL file.
| **Sources**: The OpenMDAO object responsible for recording the case. Can be `Problem`, `Driver` or a specific `System` or `Solver` identified by pathname.

Basic Recording Example
------------------------

Below is a basic example of how to create a recorder, attach it to a Problem, save the information,
and retrieve the data from the recorder. `list_outputs` is a quick way to show all of your outputs
and their values at the time the case was recorded, and should you need to isolate a single value OpenMDAO provides two ways to
retrieve them. To view all the design variables, constraints, and
objectives, you can use their methods like the example below.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_basic_case_recording
    :layout: interleave

