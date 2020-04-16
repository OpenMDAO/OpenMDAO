.. _driver_options:

*****************
Driver Recording
*****************

A :class:`CaseRecorder<openmdao.recorders.case_recorder.CaseRecorder>` is commonly attached to the
problem’s Driver in order to gain insight into the convergence of the model as the driver finds a
solution. By default, a recorder attached to a driver will record the design variables, constraints
and objectives.

Driver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.driver
    Driver
    recording_options

.. note::
    Note that the :code:`excludes` option takes precedence over the :code:`includes` option.

Driver Recording Example
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_driver_options
    :layout: interleave
