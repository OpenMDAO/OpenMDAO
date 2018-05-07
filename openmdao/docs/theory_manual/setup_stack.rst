.. _theory_setup_stack:

*********************************************************************
The System Setup Stack: Understanding When to Use setup and configure
*********************************************************************

One question that is often asked is: why can't we just put all of our model building in our group's
`__init__` method so that everything is there when I instantiate the class? The answer is, when
running a parallel model under MPI, certain systems might only be executed on certain processors.
To save memory across the model, these systems are not fully set up on processors where they are
not local. The only way to do this is to isolate the model building process into a custom method
(`setup`) and only call it on the processors where you need the footprint for that group. While
not everyone will run their models in parallel, it is a good practice to follow the stricter
guideline so that, if someone wants to include your model in a larger parallel model, they won't
be forced to refactor.

.. _theory_setup_vs_configure:

Usage of setup vs. configure
----------------------------

Here is a quick guide covering what you can do in the setup and configure methods.

**setup**

 - Add subsystems
 - Issue connections
 - Assign linear and nonlinear solvers at this group level
 - Change solver settings for any solver at this group level
 - Assign Jacobians at this group level
 - Set system execution order
 - Add desvars, objectives, and constraints
 - Add a case recorder to the group or a solver in that group.

**configure**

 - Assign linear and nonlinear solvers to subsystems
 - Change solver settings in subsystems
 - Assign Jacobians to subsystems
 - Set execution order in subsystems
 - Add a case recorder to a subsystem or a solver in a subsystem.

**Things you sould never do in configure**

 - Add subsystems
 - Delete subsystems

Problem setup and final_setup
-----------------------------

OpenMDAO 2.0 introduces a new change to the setup process in which the original monolithic process
is split into two separate phases triggered by the methods: `setup` and `final_setup`. The `final_setup` method is
however something you will probably never have to call, as it is called automatically the first time that
you call `run_model` or `run_driver` after running setup. The reason that the setup process was split into two
phases is to allow you to perform certain actions after setup:

**Post setup actions**

 - Set values of unconnected inputs and indepvarcomps
 - Change settings on solvers
 - Change options on systems
 - Add recorders
 - Assign Jacobians
 - Add training data to metamodels

If you do anything that changes the model hiearchy, such as adding a component to a group, then you will need to
run `setup` again. During setup, the following happens:

 - MPI processors are allocated
 - For each custom Group, setup function is called recursively from top to bottom
 - Model hierarchy is created
 - For each custom Group, configure function is called recursively from bottom to top
 - Variables are sized
 - Connections are assembled and verified

This is just enough to allow you to perform the post-setup actions listed above, but there are
still more things to do before the model can run. In `final_setup`, the following happens:

 - All vectors for the nonlinear and linear systems are created and allocated
 - Data transfers are created (i.e., scatters for MPI)
 - Solvers are setup
 - Jacobians are setup and allocated
 - Recorders are setup
 - Drivers are setup
 - Initial values are loaded into the inputs and outputs vectors