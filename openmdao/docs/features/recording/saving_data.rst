.. _saving_data:

**************
Case Recording
**************

Driver Recording
----------------

A :class:`CaseRecorder<openmdao.recorders.case_recorder.CaseRecorder>` is commonly attached to
the problem's :class:`Driver` in order to gain insight into the convergence of the model as the driver
finds a solution.  By default, a recorder attached to a driver will record the design variables, 
constraints and objectives.

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


Problem Recording
-----------------

You might also want to attach a recorder to the problem itself. This allows you to record an 
arbitrary case at a point of your choosing.  This feature can be useful if you only record a
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


System and Solver Recording
---------------------------

If you need to focus on a smaller part of your model, it may be useful to attach a case recorder to
a particular :code:`System` or :code:`Solver`. There are slightly different options when recording from these objects.

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
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_system_options
    :layout: interleave

Solver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.solvers.solver
    Solver
    recording_options

.. note::
    Note that the :code:`excludes` option takes precedence over the :code:`includes` option.

.. note::
    The paths given in the :code:`includes` and :code:`excludes` options are relative to the `Group` that the solver
    is attached to.

.. note::
    It is currently not possible to record linear solvers.


Solver Recording Example
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_options
    :layout: interleave


Specifying a Case Prefix
------------------------

It is possible to record data from multiple executions by specifying a prefix that will be used to
differentiate the cases.  This prefix can be specified when calling :code:`run_model` or
:code:`run_driver` and will be prepended to the case ID in the recorded case data:

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_record_with_prefix
    :layout: interleave

.. note::
    A recorder can be attached to more than one object. Also, more than one recorder can be 
    attached to an object.

.. note::
    In this example, we have disabled the saving of data needed by the standalone :math:`N^2` 
    visualizer and debugging tool by setting :code:`record_viewer_data` to :code:`False`.

Recording Options Include and Exclude Matching
----------------------------------------------

The :code:`includes` and :code:`excludes` recording options provide support for Unix shell-style wildcards,
which are not the same as regular expressions. The documentation for the :code:`fnmatchcase` function from the Python
standard library documents the wildcards: https://docs.python.org/2/library/fnmatch.html#fnmatch.fnmatchcase.

Recording Options Precedence
----------------------------

The recording options precedence that determines what gets recorded can sometime be a little confusing. Here is
an example that might help. The code shows how the :code:`record_desvars` and :code:`includes` options interact.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_recording_option_precedence
    :layout: interleave
