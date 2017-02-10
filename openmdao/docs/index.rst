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

Defining partial derivatives
================================
.. toctree::
   :maxdepth: 1

   features/defining_partials/specifying_partials

Grouping components for more complex models
================================================================

Solvers (nonlinear and linear)
================================

.. toctree::
   :maxdepth: 1

   features/solvers/set_solvers
   features/solvers/solver_options


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

.. _OpenMDAO Reference Sheets: srcdocs/index.html

`OpenMDAO Developer Source Documentation`__

.. __: srcdocs/dev/index.html

`OpenMDAO User Source Documentation`__

.. __: srcdocs/usr/index.html

.. toctree::
   :maxdepth: 1

   tags/index
