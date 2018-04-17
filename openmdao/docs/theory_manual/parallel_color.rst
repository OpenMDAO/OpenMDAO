
.. _parallel-derivatives-theory:

*******************************
Parallel Derivative Computation
*******************************

Computing derivatives with respect to multiple variables in parallel can result
in significant performance increases over computing them in serial, but this
should only be done for models where the sets of variables that are dependent
on the variables of interest have little or no overlap between them.

For example, a very simple and clear-cut case where parallel derivative
computation would be recommended is the following model, where only one constraint
is dependent on each design variable.  *Con1.y* only depends on *Indep1.x* and
*Con2.y* only depends on *Indep2.x*.  Assuming the same sizes for the design variables
and constraints, parallel derivatives would work equally well with this model in
fwd or rev mode.


.. figure:: decoupled_model.png
   :align: center
   :alt: An obvious case where parallel derivatives are appropriate.

   An obvious case where parallel derivatives are appropriate.


Often, more realistic models do not demonstrate such complete independence between
the variables of interest, but even in some of these cases, parallel derivative
computation can still be of significant benefit.  For example, suppose we have
a model where we perform some sort of preliminary calculations that don't take
vary long to run, and we feed those results into multiple components, for example CFD
components, that do take a long time to run.


.. figure:: dependent_model.png
   :align: center
   :alt: A less obvious case where parallel derivatives are appropriate.

   A less obvious case where parallel derivatives are appropriate.


In the model above, both of our constraints, *Con1.y* and *Con2.y* are dependent
on our design variable *Indep1.x*.  Let's assume here also that the size of our
design variable is the same as the combined size of our constraints, and that
*Con1* and *Con2* take much longer to run than *Comp1*.
If we solve for our derivatives using adjoint (rev) mode and we group *Con1.y* and
*Con2.y* by specifying that they have the same *parallel_deriv_color*, we will
compute derivatives for *Con1* and *Con2* concurrently while solving for
the derivatives of *Con1.y wrt Indep1.x* and *Con2.y wrt Indep1.x*.  This will
require that the *Comp1* derivative computation is
duplicated in each process, but we don't care since it's fast compared
to *Con1* and *Con2*.


The code below defines the model described above:

.. embed-code::
      openmdao.core.tests.test_parallel_derivatives.PartialDependGroup


And here we see that rev mode with parallel derivatives is roughly twice as fast
as fwd mode when our 'slow' components have a delay of .1 seconds.  Without parallel
derivatives, the fwd and rev speeds are roughly equivalent.

.. embed-code::
    openmdao.core.tests.test_parallel_derivatives.ParDerivColorFeatureTestCase.test_fwd_vs_rev
    :layout: interleave
