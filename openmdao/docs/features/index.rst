:orphan:

********
Features
********

OpenMDAO's fully supported features are documented here.
Anything documented here has been thoroughly tested an should be considered fully functional.


Capabilities
============

Components
----------
.. toctree::
   :maxdepth: 1

   building_components/building_components
   building_components/declaring_variables
   recording/basic_recording

Design Variables, Objectives, and Constraints
---------------------------------------------
.. toctree::
   :maxdepth: 1

   drivers/add_vois


Grouping components for more complex models
-------------------------------------------
.. toctree::
   :maxdepth: 1

   grouping_components/grouping_components
   grouping_components/add_subsystem
   grouping_components/get_subsystem
   grouping_components/connect
   grouping_components/set_order
   grouping_components/src_indices
   grouping_components/parallel_group
   grouping_components/standalone_groups


Running your models
-------------------
.. toctree::
   :maxdepth: 1

   running/set_get
   running/listing_variables
   running/setup_and_run
   running/check_total_derivatives

Units and Scaling
-----------------
.. toctree::
   :maxdepth: 1

   building_components/units
   building_components/scaling

Controlling Solver Behaviors
----------------------------
.. toctree::
   :maxdepth: 1

   solvers/set_solvers
   solvers/solver_options

Specialized components
----------------------
.. toctree::
   :maxdepth: 1

   special_components/distributed_comps
   special_components/metamodel

Defining partial derivatives
----------------------------
.. toctree::
   :maxdepth: 1

   defining_partials/specifying_partials
   defining_partials/sparse_partials
   defining_partials/approximating_partials
   defining_partials/checking_partials


Building Blocks
===============
.. toctree::
   :maxdepth: 1

Solvers (nonlinear and linear)
------------------------------
.. toctree::
   :maxdepth: 1

   solvers/nonlinear_runonce
   solvers/nonlinear_block_gs
   solvers/nonlinear_block_jac
   solvers/newton
   solvers/linear_runonce
   solvers/direct_solver
   solvers/linear_block_gs
   solvers/scipy_iter_solver
   solvers/petsc_ksp
   solvers/linear_block_jac
   solvers/linesearch_backtracking

Drivers
-------
.. toctree::
   :maxdepth: 1

   drivers/scipy_optimizer

Components
----------
.. toctree::
   :maxdepth: 1

   building_components/indepvarcomp
   building_components/explicitcomp
   building_components/implicitcomp


Visualization and Analysis
==========================
.. toctree::
   :maxdepth: 1

   recording/basic_recording
   devtools/inst_profile