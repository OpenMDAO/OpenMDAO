.. _solver_options:

*****************
Solver Recording
*****************

Solver recording is useful when you want to record the iterations within a solver.
The recorder can capture the values of states, errors, and residuals as the solver converges.

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
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_solver_recording_options
    :layout: interleave

