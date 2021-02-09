.. _driver_options:

*****************
Driver Recording
*****************

A :class:`CaseRecorder<openmdao.recorders.case_recorder.CaseRecorder>` is commonly attached to the
problem’s Driver in order to gain insight into the convergence of the model as the driver finds a
solution. By default, a recorder attached to a driver will record the design variables, constraints
and objectives.

The driver recorder is capable of capturing any values from any part of the model,
not just the design variables, constraints, and objectives.

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
In the example below, we first run a case while recording at the driver level.
Then, we examine the objective, constraint, and design variable values at the
last recorded case.
Lastly, we print the full contents of the last case, including outputs from
the problem that are not design variables, constraints, or objectives. 
Specifically, `y1` and `y2` are some of those intermediate outputs that are recorded due to the use of :code:`driver.recording_options['includes'] = ['*']` above.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_driver_recording_options
    :layout: interleave
