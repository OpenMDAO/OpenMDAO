.. _feature_simul_coloring_approx:


*************************************************
Simultaneous Coloring of Approximated Derivatives
*************************************************


In OpenMDAO, partial derivatives for components and semi-total derivatives for groups can
be approximated using either finite difference or complex step.  Sometimes the partial or
semi-total jacobians in these cases are sparse, and the computing of these jacobians can
be made more efficient using simultaneous derivative coloring.  For an explanation of a
similar coloring for total derivatives, see
:ref:`Simultaneous Coloring For Separable Problems<feature_simul_coloring>`.  Finite difference
and complex step only work in forward mode, so only a forward mode coloring is possible when
using them, but depending on the sparsity pattern of the jacobian, it may still be possible
to get significant efficiency gains.

Setting up a problem to use simultaneous coloring of approximated derivatives requires a
call to the :code:`set_approx_coloring` function.  For example, the code below sets up coloring for
partial derivatives of outputs of `comp` with respect to inputs of `comp` starting with 'x'.
Let's assume here that :code:`MyComp` is an :code:`ExplicitComponent`.  If it were an
:code:`ImplicitComponent`, then the wildcard pattern 'x*' would be applied to all inputs *and*
outputs (states) of `comp`.

.. code-block:: python

    comp = prob.model.add_subsystem('comp', MyComp(dynamic_derivs_repeats=2))
    comp.set_approx_coloring('x*', method='cs')


Note that in addition to the call to :code:`set_approx_coloring`, we also set
:code:`dynamic_derivs_repeats` to 2 when we instantiate 'comp'.  This means that the first
2 times that a partial jacobian is computed for 'comp', it's values will be computed without
coloring and stored.  Just prior to the 3rd time, the coloring will be computed and used for
the rest of the run.

Semi-total derivative coloring can be performed in a similar way, except that
:code:`set_approx_coloring` would be called on a :code:`Group` instead of a :code:`Component`.
:code:`dynamic_derivs_repeats` can also be set on the :code:`Group`.


