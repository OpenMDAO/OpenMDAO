
**********************************
Vectorizing Derivative Computation
**********************************

To turn on vectorized derivative computation, set the *vectorize_derivs* arg to True
when calling *add_design_var* in fwd mode or *add_constraint* and/or *add_objective*
in rev mode.

For example, in fwd mode:

.. code-block:: python

    model.add_design_var('y_lgl', lower=-1000.0, upper=1000.0, vectorize_derivs=True)


In rev mode:

.. code-block:: python

    model.add_constraint('defect.defect', lower=-1e-6, upper=1e-6, vectorize_derivs=True)
