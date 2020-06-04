.. _system_options:

*****************
System Recording
*****************

System Recording
---------------------------

If you need to focus on a smaller part of your model, it may be useful to attach a case recorder to
a particular :code:`System`. There are slightly different options when recording from these objects.

System Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.system
    System
    recording_options

.. note::
    Note that the :code:`excludes` option takes precedence over the :code:`includes` option.

System Recording Example
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_system_recording_options
    :layout: interleave