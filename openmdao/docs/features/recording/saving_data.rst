.. _saving_data:

**********************************
Saving Data with the Case Recorder
**********************************

In OpenMDAO, you can instantiate recorder objects and attach them to the System, Driver or Solver
instance(s) of your choice.

Instantiating a Recorder
++++++++++++++++++++++++

Instantiating a recorder is easy.  Simply give it a name, choose which type of recorder you want (currently only
SqliteRecorder exists), and name the output file that you would like to write to.

.. code-block:: console

    my_recorder = SqliteRecorder("filename")

.. note::
    Currently, appending to an existing DB file is not supported; the SQLite recorder
    will automatically write over an existing file if it carries the same name.


Setting Recording Options
+++++++++++++++++++++++++

There are many recording options that can be set. This affects the amount of information retained by the recorders.
These options are associated with the System, Driver or Solver that is being recorded.


The following examples use the :ref:`Sellar <sellar>` model and demonstrate setting the recording options on
each of these types.


Recording on System Objects
---------------------------

System Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.system
    System
    recording_options

To record on a System object, simply add the recorder to the System and set the recording options.
Note that the 'excludes' option takes precedence over the 'includes' option, as shown in the example
below where we exclude `obj_cmp.x` and see that it isn't in the set of recorded inputs.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_system_options
    :layout: interleave

Recording on Driver Objects
---------------------------

Driver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.driver
    Driver
    recording_options

Recording on a Driver is very similar to recording on a System, though it has a few additional recording options.
The options 'record_objectives', 'record_constraints', 'record_desvars', and 'record_responses' are all still limited by
'excludes', but they do take precedence over 'includes', as shown below where 'includes'
is empty but objectives, constraints, and desvars are still recorded.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_driver_options
    :layout: interleave

Recording on Solver Objects
---------------------------

Solver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.solvers.solver
    Solver
    recording_options

Recording on Solvers is nearly identical to recording on Systems with the addition of options for recording absolute and relative
error. Below is a basic example of adding a recorder to a solver object and recording absolute error.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_options
    :layout: interleave

.. note::
    A recorder can be attached to more than one object. Also, more than one recorder can be attached to an object.


Recording on Problem Objects
----------------------------

Problem Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.problem
    Problem
    recording_options

Recording on Problems is different from recording other objects because nothing is recorded automatically. The
user must explicitly call the `Problem.record_iteration` method to record the current values from the `Problem`.
Below is a basic example of adding a recorder to a `Problem` object and then recording it after a run.

This feature can be useful if you only record a limited number of variables during the run but would like to see a more
complete list of values after the run.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_problem_record
    :layout: interleave



Specifying a Case Prefix
------------------------

It is possible to record data from multiple executions by specifying a prefix that will be used to differentiate the
cases.  This prefix can be specified when calling `run_model` or `run_driver` and will be prepended to the case ID
in the recorded case data:


.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_record_with_prefix
    :layout: interleave
