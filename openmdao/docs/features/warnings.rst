.. _feature_warnings:

***************
Warning Control
***************

OpenMDAO has several classes of warnings that may be raised during operation.
In general, these warnings are useful and the user should pay attention to them.
Sometimes these warnings can be unnecessarily noisy.
Filtering out noisy "low-priority" warning can make other more important ones more obvious.

OpenMDAO-Specific Warning Categories
------------------------------------

Class **OpenMDAOWarning** serves as the base-class for all OpenMDAO-specific warnings.
All OpenMDAO-specific warnings default to a filter of 'always'.
The following table shows all OpenMDAOWarning-derived classes.

+-----------------------------+---------------------+----------------------+-----------------------------------------------+
| Warning Class               | String Name         | Default Behavior     | Description                                   |
+=============================+=====================+======================+===============================================+
| CacheWarning                | A cache is invalid and must be discarded.                                                  |
| CaseRecorderWarning         | Warning encountered by a case recorder or case reader.                                     |
| DerivativesWarning          | Warning issued when the approximated partials or coloring cannot be evaluated as expected. |
| DriverWarning               | Warning encountered during driver execution.                                               |
| OMDeprecationWarning        | Warning of the use of deprecated OpenMDAO features.                                        |
| SetupWarning                | General errors encountered during setup.                                                   |
| SolverWarning               | Warning encountered during solver execution.                                               |
| UnusedOptionWarning         | A given option or argument has no effect.                                                  |
+-----------------------------+---------------------+----------------------+-----------------------------------------------+

Note that the OpenMDAO-Specific **OMDeprecationWarning** behaves a bit differently than the default Python DeprecationWarning.
**OMDeprecationWarning** is is always displayed by default, but can be silenced by the user.

For finer control over which warnings are displayed during setup, the following warning classes derive from **SetupWarning**.
Using a filter to silence SetupWarning will silence **all** of the following.

+-----------------------------+--------------------------------------------------------------------------------------------+
| Option Name                 | Description                                                                                |
+=============================+=====================+======================================================================+
| DistributedComponentWarning | Warning specific to a distributed component.                                               |
| MPIWarning                  | Warning dealing with the availability of MPI.                                              |
| PromotionWarning            | Warning dealing with the promotion of an input or output.                                  |
| SetupWarning                | Warning class for warnings that occur during setup.                                        |
| UnitsWarning                | Warning which is issued when unitless variable is connected to a variable with units.      |
+-----------------------------+--------------------------------------------------------------------------------------------+

Filtering Warnings
------------------

Python's built-in warning filtering system can be used to control which warning are displayed when using OpenMDAO.
The following script generates an OpenMDAO model which will generate UnitsWarning due to connecting unitless outputs to inputs with units.

In the following code, the UnitsWarning will be displayed as expected:

.. embed-code::
    openmdao.test_suite.tests.test_warnings.TestWarnings.test_doc_with_units_warning
    :layout: code, output

The warnings can be completely turned off by filtering them using Python's `filterwarnings` function:

.. embed-code::
    openmdao.test_suite.tests.test_warnings.TestWarnings.test_doc_ignore_units_warning
    :layout: code, output

If you want to clean your code and remove warnings, it can be useful to promote them to errors so that they cannot be ignored.
The following code filters **all** OpenMDAO associated warnings to Errors:

.. embed-code::
    openmdao.test_suite.tests.test_warnings.TestWarnings.test_doc_error_on_openmdao_warning
    :layout: code, output


Notes for Developers
--------------------

Python's treatment of warnings inside UnitTest tests can be somewhat confusing.
If you wish to test that certain warnings are filtered during testing, we recommend using the `om.reset_warnings()` method in the `setUp` method that is run before each test in a `TestCase`.

.. code::

    import unittest
    import openmdao.api as om

    class MyTestCase(unittest.TestCase):

        def setUp(self):
            """
            Ensure that OpenMDAO warnings are using their default filter action.
            """
            om.reset_warnings()

        def test_a(self):
            ...
