.. _basic_recording:

**************************
Recording and Reading Data
**************************

One of the most useful features in OpenMDAO is :ref:`Case Recording <case_recording>`. 
With case recording you can record variables and their values over the course of an optimization
and store the information in a local database file.  The data can then be read back later using
a case reader.

This can be used for debugging, plotting/visualization, and even starting a new optimization
where an old one left off.


Creating and Attaching a Recorder
---------------------------------

In the following example we add a recorder to the :ref:`Sellar<sellar>` Problem:

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureBasicRecording.record_cases
    :layout: code

Note that cases haved been saved to a database file named `cases.sql`.


Creating a Case Reader and Reading Data
---------------------------------------

After the optimization has been run, we can read the data from the file using a case reader:

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureBasicRecording.test_read_cases
    :layout: interleave
