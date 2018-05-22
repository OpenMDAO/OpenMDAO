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


Setting Recording Options
+++++++++++++++++++++++++

There are many recorder options that can be set. This affects the amount of information retained by the recorders.
These options are associated with the System, Driver or Solver that is being recorded.

A basic example of how to set an option:

.. code-block:: console

    prob.driver.recording_options['record_desvars'] = True

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

Recording on Solvers is nearly identical to recording on Systems with the additon of options for recording absolute and relative
error. Below is a basic example of adding a recorder to a solver object and recording absolute error.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_options
    :layout: interleave

.. note::
    A recorder can be attached to more than one object. Also, more than one recorder can be attached to an object.
