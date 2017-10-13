**************************************
Building Models with Standalone Groups
**************************************

So far we have showed how to build groups, add subsystems to them, and connect variables in your run-script. You can also
create custom groups that can be instantiated as part of larger models. To do this, create a new class that inherits from
Group and give it one method named "setup". Inside this method, you can add subsystems, make connections, and modify the
group level linear and nonlinear solvers.

Here is an example where we take two Sellar models and connect them in a cycle. We also set the linear and nonlinear solvers.

.. embed-code::
    openmdao.test_suite.components.sellar_feature.DoubleSellar

Configuring Child Subsystems
----------------------------

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

Here is a reference on what you can do in each method:

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

Post Setup Configuration
------------------------

OpenMDAO allows you to do a limited number of things after setup is called. These include the following:

 - Set initial conditions for unconnected inputs or states
 - Assign linear and nonlinear solvers
 - Change solver settings
 - Assign Jacobians
 - Set execution order
 - Assign case recorders

In previous versions of OpenMDAO, it was not possible to do these things after setup (with the exception of setting initial conditions.)
In OpenMDAO 2.0, the setup process was redesigned and split into two phases. In the first phase, executed when `setup` is called, the
model hiearchy is assembled, the processors are allocated (for MPI), and variables and connections are all assigned. At this point,
subsystems are accessible as attributes on their parent. The second phase of setup runs automatically when you call `run_driver` or
`run_model`. During this phase, the vectors are created and populated, the drivers and solvers are initialized, and the recorders are
started.

Here, we instantiate a hierarchy of groups, and then change the solver to one that can solve this problem.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_post_setup_solver_configure

.. tags:: Group, System