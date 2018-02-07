.. _feature_configure:

************************************************
Modify Children of a Group with Configure Method
************************************************


Most of the time, the :code:`setup` method is the only one you need to define on a group. The main exception is the case where you
want to modify a solver that was set in one of your children groups. When you call :code:`add_subsystem`, the system you add is
instantiated, but its :code:`setup` method is not called until after the parent group's :code:`setup` method is finished with its
execution. That means that anything you do with that subsystem (e.g., changing the nonlinear solver) will potentially be
overwritten by the child system's :code:`setup` if it is assigned there as well.

To get around this timing problem, there is a second setup method called :code:`configure`, that runs after the :code:`setup` on all
subsystems has completed. While :code:`setup` recurses from the top down, :code:`configure` recurses from the bottom up, so that the highest
system in the hierarchy takes precedence over all lower ones for any modifications.

Here is a simple example where a lower system sets a solver, but we want to change it to a different one in the top-most
system.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_system_configure
    :no-split:


Uses of setup vs. configure
---------------------------

**setup**

 - Add subsystems
 - Issue connections
 - Assign linear and nonlinear solvers at group level
 - Change solver settings in group
 - Assign Jacobians at group level
 - Set system execution order

**configure**

 - Assign linear and nonlinear solvers to subsystems
 - Change solver settings in subsystems
 - Assign Jacobians to subsystems
 - Set execution order in subsystems

