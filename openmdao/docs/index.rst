============================
`Tutorials`_
============================
.. _OpenMDAO Tutorials: tutorials/index.html

These tutorials serve to give an initial introduction to the OpenMDAO framework.
They are intended to be done in sequence. Once completed,
you will be able to construct models by linking multiple analyses together and then perform optimization on those models.

.. toctree::
   :maxdepth: 1

   tutorials/first_analysis
   tutorials/first_optimization
   tutorials/first_mdao


===============================
`Sample Problems`_
===============================

.. toctree::
   :maxdepth: 1

   features/problems/sellar

============================
`Features`_
============================
.. _OpenMDAO Features: features/index.html

OpenMDAOs fully supported features are documented here.
Anything documented here has been thoroughly tested an should be considered fully functional.



Building components
====================

.. toctree::
   :maxdepth: 1

   features/building_components/building_components
   features/building_components/declaring_variables
   features/building_components/indepvarcomp
   features/building_components/explicitcomp
   features/building_components/implicitcomp
   features/building_components/units
   features/building_components/scaling

Grouping components for more complex models
===========================================

.. toctree::
    :maxdepth: 1

    features/grouping_components/grouping_components
    features/grouping_components/add_subsystem
    features/grouping_components/get_subsystem
    features/grouping_components/connect
    features/grouping_components/set_order
    features/grouping_components/src_indices
    features/grouping_components/parallel_group
    features/grouping_components/standalone_groups

Defining partial derivatives
================================
.. toctree::
   :maxdepth: 1

   features/defining_partials/specifying_partials
   features/defining_partials/sparse_partials
   features/defining_partials/approximating_partials
   features/defining_partials/checking_partials

Solvers (nonlinear and linear)
================================

.. toctree::
   :maxdepth: 1

   features/solvers/set_solvers
   features/solvers/solver_options
   features/solvers/nonlinear_runonce
   features/solvers/nonlinear_block_gs
   features/solvers/nonlinear_block_jac
   features/solvers/newton
   features/solvers/linear_runonce
   features/solvers/direct_solver
   features/solvers/linear_block_gs
   features/solvers/scipy_iter_solver
   features/solvers/petsc_ksp
   features/solvers/linear_block_jac
   features/solvers/linesearch_backtracking

Drivers
=======

.. toctree::
   :maxdepth: 1

   features/drivers/add_vois
   features/drivers/scipy_optimizer


Running your models
================================

.. toctree::
   :maxdepth: 1

   features/running/set_get
   features/running/setup_and_run
   features/running/check_total_derivatives

Specialized components
======================

.. toctree::
   :maxdepth: 1

   features/special_components/distributed_comps
   features/special_components/metamodel


Development Tools
=================

.. toctree::
   :maxdepth: 1

   features/devtools/inst_profile


Drivers (optimizers and DOE)
=============================

Saving your data
================

Visualization
=============

============================
`Source Documentation`_
============================

.. _OpenMDAO Reference Sheets: _srcdocs/index.html

`OpenMDAO User Source Documentation`__

.. __: _srcdocs/usr/index.html

.. toctree::
   :maxdepth: 1

   tags/index

Style for Developers
====================

.. toctree::
   :maxdepth: 1

   style_guide/doc_style_guide.rst
