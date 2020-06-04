.. _feature_simul_coloring_approx:


*************************************************
Simultaneous Coloring of Approximated Derivatives
*************************************************


In OpenMDAO, partial derivatives for components and semi-total derivatives for groups can
be approximated using either finite difference or complex step.  Sometimes the partial or
semi-total jacobians in these cases are sparse, and the computing of these jacobians can
be made more efficient using simultaneous derivative coloring.  For an explanation of a
similar coloring for *total* derivatives, see
:ref:`Simultaneous Coloring For Separable Problems<feature_simul_coloring>`.  Finite difference
and complex step only work in forward mode, so only a forward mode coloring is possible when
using them, but depending on the sparsity pattern of the jacobian, it may still be possible
to get significant efficiency gains.


Dynamic Coloring
================

Setting up a problem to use dynamic coloring of approximated derivatives requires a
call to the :code:`declare_coloring` function.


.. automethod:: openmdao.core.system.System.declare_coloring
    :noindex:


For example, the code below sets up coloring for
partial derivatives of outputs of `comp` with respect to inputs of `comp` starting with 'x'.
Let's assume here that :code:`MyComp` is an :code:`ExplicitComponent`.  If it were an
:code:`ImplicitComponent`, then the wildcard pattern 'x*' would be applied to all inputs *and*
outputs (states) of `comp`.

.. code-block:: python

    comp = prob.model.add_subsystem('comp', MyComp())
    comp.declare_coloring('x*', method='cs', num_full_jacs=2, min_improve_pct=10.)


Note that in the call to :code:`declare_coloring`, we also set :code:`num_full_jacs` to 2.  This means
that the first 2 times that a partial jacobian is computed for 'comp', it's values will be computed
without coloring and stored.  Just prior to the 3rd time, the coloring will be computed and used for
the rest of the run.  We also set :code:`min_improve_pct` to 10, meaning that if the computed
coloring does not reduce the number of nonlinear solves required to compute `comp's` partial jacobian,
then `comp` will not use coloring at all.

Semi-total derivative coloring can be performed in a similar way, except that
:code:`declare_coloring` would be called on a :code:`Group` instead of a :code:`Component`.
:code:`num_full_jacs` can also be passed as an arg to :code:`declare_coloring` on the :code:`Group`.

The purpose of :code:`declare_coloring` is to provide all of the necessary information to allow
OpenMDAO to generate a coloring, either dynamically or manually using :code:`openmdao partial_coloring`.

Coloring files that are generated dynamically will be placed in the directory specified in
:code:`problem.options['coloring_dir']` and will be named based on the value of the
:code:`per_instance` arg passed to :code:`declare_coloring`.  If :code:`per_instance` is True,
the file will be named based on the full pathname of the component or group being colored.  If
False, the name will be based on the full module pathname of the class of the given
component or group.

:code:`declare_coloring` should generally be called in the :code:`setup` function of the
component or group.

Here's a modified version of our total coloring example, where we replace one of our components
with one that computes a dynamic partial coloring.  A total coloring is also performed, as in the
previous example, but this time the total coloring uses sparsity information computed by our
component during its dynamic partial coloring.



.. embed-code::
    openmdao.core.tests.test_coloring.SimulColoringScipyTestCase.test_total_and_partial_coloring_example
    :layout: interleave
    :imports-not-required:


Static Coloring
===============

Static partial or semi-total derivative coloring is activated by calling the
:code:`use_fixed_coloring` function on the corresponding component or group, after
calling :code:`declare_coloring`.

.. automethod:: openmdao.core.system.System.use_fixed_coloring
    :noindex:

Generally, no arg will be passed to :code:`use_fixed_coloring`, and OpenMDAO will automatically
determine the location and name of the appropriate coloring file, but it is possible to pass
the name of a coloring file into :code:`use_fixed_coloring`, and in that case the given
coloring file will be used.  Note that if a coloring filename is passed into :code:`use_fixed_coloring`,
it is assumed that the coloring in that file should *never* be regenerated, even if the user
calls :code:`openmdao total_coloring` or :code:`openmdao partial_coloring` from the command line.

