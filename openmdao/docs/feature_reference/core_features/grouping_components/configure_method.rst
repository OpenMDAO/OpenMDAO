.. _feature_configure:

*********************************************************
Modify Children of a Group with Configure Method
*********************************************************


Most of the time, the "setup" method is the only one you need to define on a group. The main exception is the case where you
want to modify a solver that was set in one of your children groups. When you call add_subsystem, the system you add is
instantiated, but its "setup" method is not called until after the parent group's "setup" method is finished with its
execution. That means that anything you do with that subsystem (such as changing the nonlinear solver) will potentially be
overwritten by the child system's setup if it is assigned there as well.

To get around this timing problem, there is a second setup method called "configure" that runs after the "setup" on all
subsystems has completed. While "setup" recurses from top down, "configure" recurses from bottom up, so that the highest
system in the hiearchy takes precedence over all lower ones for any modifications.

Here is a simple example where a lower system sets a solver, but we want to change it to a different one in the top-most
system.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_system_configure
    :no-split:



What you can do with each method
----------------------------------

**setup**

 - Add subsystems
 - Issue Connections
 - Assign linear and nonlinear solvers at group level
 - Change solver settings in group
 - Assign Jacobians at group level
 - Set system execution order

**configure**

 - Assign linear and nonlinear solvers to subsystems
 - Change solver settings in subsystems
 - Assign Jacobians to subsystems
 - Set execution order in subsystems

