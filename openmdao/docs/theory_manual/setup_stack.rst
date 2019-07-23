.. _theory_setup_stack:

*********************************************************************
The System Setup Stack: Understanding When to Use setup and configure
*********************************************************************

This document explains what happens during the OpenMDAO `Problem` `setup` process, and how some of the model
API methods interact during that process.

The purpose of the `setup` process is to prepare the data structures that OpenMDAO needs to efficiently
run your model or driver. In particular, this includes setting up the vectors used for passing data
to inputs, converging solvers, and calculating derivatives. It also includes setting up the MPI
communicators.

Setup also performs some level of model checking, mainly for critical errors. More extensive model
checking can be done by setting "check" when calling `setup`, or by using the :ref:`openmdao command
line check<om-command>`. It is recommended that you do this after making any changes to the configuration
of your model.  The "check" argument to `setup` can be set to `True`, which will cause a default
set of checks to run.  It can also be set to 'all', which will run all available checks.  Finally,
it can be set to a specific list of checks to run.  The checks that are available can be
determined by running the following command:

.. embed-shell-cmd::
    :cmd: openmdao check -h

By default, the output of all checks will be written to a file called 'openmdao_checks.out' in
addition to `stdout`.  Checks can also be performed by calling the `check_config` method on
your problem object.


The OpenMDAO `Group` API includes three methods that are invoked during the `setup` process: `setup`, `configure`, and
`initialize`. Most of the time, `setup` is all you need to build a group. The specific use case for
`configure` is shown below in the next section. The `initialize` method is only used for declaring options for your
group (and also in `Component`), and their placement here allows them to be passed into the group as
instantiation arguments.

One question that is often asked is: why can't we just put all of our model building code into our group's
`__init__` method so that everything is there when I instantiate the class? The answer is, when
running a parallel model under MPI, certain systems might only be executed on certain processors.
To save memory across the model, these systems are not fully set up on processors where they are
not local. The only way to do this is to isolate the model building process into a custom method
(`setup`) and only call it on the processors where you need the footprint for that group. While
not everyone will run their models in parallel, it is a good practice to follow the stricter
guideline so that, if someone wants to include your model in a larger parallel model, they won't
be forced to refactor it.

.. _theory_setup_vs_configure:

Usage of setup vs. configure
----------------------------

The need for two methods for setting up a group arose from a need to sometimes change the linear or
nonlinear solvers in a subgroup after it has been added. When `setup` is called on the `problem`, the
`setup` method in each group is called recursively from top to bottom of the hierarchy. For example,
a group may contain several components and groups. Setup is first called in that top group, during
which, those components and groups are instantiated. However, the `setup` methods belonging to those sub-components
and groups cannot be called until the top group's `setup` finishes. This means they are in a state where
components and groups that are declared in the subgroup don't exist yet.

To remedy this, there is a second api method called `configure` that lets you make changes to your subsystems
after they have been created. The `configure` method is only needed with groups, and it is called
recursively from the bottom of the hierarchy to the top, so that at any level, you can be sure that
`configure` has already run for all your subsystems. This assures that changes made in higher-level groups
take precedence over those in lower-level ones. Top precedence is given to changes made after calling `setup`
on the `Problem`.

A second use case for `configure` is issuing connections to subsystems when you need information (e.g. path names)
that has been set during setup of those subsystems.  Since `configure` runs after `setup` has been
called on all subsystems, you can be sure that this information will be available.

Here is a quick guide covering what you can do in the `setup` and `configure` methods.

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

 - Issue connections
 - Assign linear and nonlinear solvers to subsystems
 - Change solver settings in subsystems
 - Assign Jacobians to subsystems
 - Set execution order in subsystems
 - Add a case recorder to a subsystem or a solver in a subsystem.

**Things you should never do in configure**

 - Add subsystems
 - Delete subsystems

 Keep in mind that, when `configure` is being run, you are already done calling `setup` on every group
 and component in the model, so if you add something here, setup will never be called, and it will
 never be fully integrated into the model hierarchy.

Problem setup and final_setup
-----------------------------

OpenMDAO 2.0 introduces a new change to the setup process in which the original monolithic process
is split into two separate phases triggered by the methods: `setup` and `final_setup`. The `final_setup` method is
however something you will probably never have to call, as it is called automatically the first time that
you call `run_model` or `run_driver` after running `setup`. The reason that the `setup` process was split into two
phases is to allow you to perform certain actions after `setup`:

**Post-setup actions**

 - Set values of unconnected inputs and indepvarcomps
 - Change settings on solvers
 - Change options on systems
 - Add recorders
 - Assign Jacobians
 - Add training data to metamodels

If you do anything that changes the model hierarchy, such as adding a component to a group, then you will need to
run `setup` again.

During setup, the following things happen:

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
 - Solvers are set up
 - Jacobians are set up and allocated
 - Recorders are set up
 - Drivers are set up
 - Initial values are loaded into the inputs and outputs vectors