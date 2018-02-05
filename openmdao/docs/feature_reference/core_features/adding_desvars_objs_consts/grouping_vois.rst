******************************************************
Grouping Design Variables, Constraints, and Objectives
******************************************************

Sometimes, based on data dependencies in a model, it's possible to solve for
derivatives with respect to certain groups of variables concurrently.  For a
detailed explanation of when it's worthwhile to do this, see :ref:`parallel-derivatives-theory`.
We can specify groups of variables that should have their derivatives computed concurrently
using the *parallel_deriv_color* argument to `add_design_var`, `add_constraint`,
or `add_objective`.  The types of variables to be grouped depend upon the direction
of derivative computation that we specified in the call to `setup()`.  In fwd mode,
we group design variables, for example:

.. code-block:: python

    model.add_design_var('y_lgl', lower=-1000.0, upper=1000.0, parallel_deriv_color='par_dvs')


In rev mode, we group responses, i.e., objectives and constraints, for example:

.. code-block:: python

    model.add_constraint('defect.defect', lower=-1e-6, upper=1e-6, parallel_deriv_color='par_cons')


Any variables that share the same `parallel_deriv_color` will be grouped during derivative
computation.
