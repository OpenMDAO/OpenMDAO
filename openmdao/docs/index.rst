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


Defining partial derivatives
================================

Solvers (nonlinear and linear)
================================

.. toctree::
   :maxdepth: 1

   features/solvers/set_solvers
   features/solvers/solver_options
   features/solvers/nl_runonce
   features/solvers/nl_bgs
   features/solvers/nl_bjac
   features/solvers/nl_newton
   features/solvers/ln_runonce
   features/solvers/ln_direct
   features/solvers/ln_bgs
   features/solvers/ln_scipy
   features/solvers/ln_petscksp
   features/solvers/ln_bjac
   features/solvers/ls_backtracking

Running your models
================================

.. toctree::
   :maxdepth: 1

   features/running/set_get
   features/running/setup_and_run
   features/running/check_total_derivatives

Drivers (optimizers and DOE)
================================

Saving your data
================================

Visualization
================================

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
