.. _saving_data:

***********************
Case Recorder Settings
***********************

Table of CaseRecorder Options
----------------------------------------------------------------
Below is a table of options that can be applied to the user's CaseRecorder

.. csv-table:: Recorder Options Table
   :header: "Record Options", "Driver", "System", "Solver", "Problem"
   :widths: 25, 10, 10, 10, 10

   "record_constraints", "X", "", "", "X"
   "record_desvars", "X", "", "", "X"
   "record_objectives", "X", "", "", "X"
   "record_derivatives", "X", "", "", "X"
   "record_responses", "X", "", "", "X"
   "record_inputs", "X", "X", "X", "X"
   "record_outputs", "X", "X", "X", "X"
   "record_residuals", "X", "X", "", "X"
   "record_abs_error", "", "", "X", ""
   "record_rel_error", "", "", "X", ""
   "record_solver_residuals", "", "", "X", ""
   "includes", "X", "X", "X", "X"
   "excludes", "X", "X", "X", "X"
   "options_excludes", "", "X", "", ""

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
standard library documents the `wildcards <https://docs.python.org/3.8/library/fnmatch.html#fnmatch.fnmatchcase>`_.

Recording Options Precedence
----------------------------

The recording options precedence that determines what gets recorded can sometime be a little confusing. Here is
an example that might help. The code shows how the :code:`record_desvars` and :code:`includes` options interact.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestFeatureSqliteReader.test_feature_recording_option_precedence
    :layout: interleave
