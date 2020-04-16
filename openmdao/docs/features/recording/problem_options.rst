.. _problem_options:

*****************
Problem Recording
*****************

You might also want to attach a recorder to the :class:`Problem` itself. This allows you to record
an arbitrary case at a point of your choosing. This feature can be useful if you only record a
limited number of variables during the run but would like to see a more complete list of values
after the run.

The options are a subset of those for driver recording.

Problem Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.problem
    Problem
    recording_options

.. note::
    Note that the :code:`excludes` option takes precedence over the :code:`includes` option.

Problem Recording Example
^^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_problem_record
    :layout: interleave