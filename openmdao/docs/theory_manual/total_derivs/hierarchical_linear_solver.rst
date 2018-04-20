******************************************************************
Using the Model Hierarchy to Customize the Linear Solver Structure
******************************************************************

In OpenMDAO, your model is constructed via collections of Groups and Components arranged hierarchically.
One of the main purposes of the hierarchy is to provide a means of sub-dividing a large and complex model into parts that can be solved using different methods.
This creates a hierarchical solver architecture that is potentially both more efficient and more effective.
The hierarchical solver architecture can be used for both nonlinear and linear solvers, but this section focuses specifically on the linear solver.

A Very Simple Example
---------------------

Consider, as an example, the :ref:`Sellar Problem<sellar>` from the :ref:`Multidisciplinary Optimization User Guide <user_guide_multi_disciplinary_opt>`.
In that problem, coupling is created by a cyclic connection between the :code:`d1` and :code:`d2` components.
You can see that coupling clearly in the n2 diagram below, because there are off-diagonal terms both above and below the diagonal inside the :code:`cycle` group.

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarMDALinearSolver

.. raw:: html
    :file: sellar_n2.html

Since there is coupling in this model, there must also be some linear solver there to deal with it.
One option would be to assign the :ref:`DirectSolver <directsolver>` right at the top level of the model, and have it compute an inverse of the full Jacobian.
While that would certainly work, you're taking an inverse of a larger matrix than you really need to.

Instead, as we've shown in the code above, you can assign the :ref:`DirectSolver <directsolver>` at the :code:`cycle` level instead.
The top level of the hierarchy will then be left with the default :ref:`LinearRunOnce<lnrunonce>` solver in it.
Effectively, the direct solver is being used to compute the coupled semi-total derivatives across the :code:`cycle` group,
which then makes the top level of the model have a feed-forward data path that can be solved with forward or back substitution
(depending whether you select :code:`fwd` or :code:`rev` mode).

To illustrate that visually, you can *right-click* on the cycle group in the n2 diagram above.
This will collapse the cycle group to a single box, and you will see the resulting uncoupled, upper-triangular matrix structure that results.

Practically speaking, for a tiny problem like :ref:`Sellar<sellar>` there won't be any performance difference between putting
the :ref:`DirectSolver <directsolver>` at the top, versus down in the :code:`cycle` group. However, in larger models with hundreds or
thousands of variables, the effect can be much more pronounced (e.g. if you're trying to invert a dense 10000x10000 matrix when
you could be handling only a 10x10).

More importantly, if you have models with high-fidelity codes like CFD or FEA in the hierarchy,
you simply may not be able to use a :ref:`DirectSolver <directsolver>` at the top of the model, but there may still be a
portion of the model where it makes sense. As you can see, understanding how to take advantage of the model hierarchy in
order to customize the linear solver behavior becomes more important as your model complexity increases.


A More Realistic Example
------------------------

Consider an aerostructural model of an aircraft wing comprised of a Computational Fluid Dynamics (CFD) solver, a simple
finite-element beam analysis, with a fuel-burn objective and a :math:`C_l` constraint.

In OpenMDAO the model is set up as follows:

.. figure:: aerostruct_n2.png
    :align: center
    :width: 75%

    :math:`N^2` diagram for an aerostructural model with linear solvers noted in :code:`()`.

Note that this model has almost the exact same structure in its :math:`N^2` diagram as the sellar problem.
Specifically the coupling between the aerodynamics and structural analyses can be isolated from the rest of the model.
Those two are grouped together in the :code:`aerostruct_cycle` group, giving the top level of the model a feed-forward structure.
There is a subtle difference though; the Sellar problem is constructed of all explicit components but this aerostructural problem has two implicit analyses in the :code:`aero` and :code:`struct` components.
Practically speaking, the presence of a CFD component means that the model is too big to use a :ref:`DirectSolver <directsolver>` at the top level of its hierarchy.

Instead, based on the advice in the :ref:`Theory Manual entry on selecting which kind of linear solver to use<theory_selecting_linear_solver>`,
the feed-forward structure on the top level indicates that the default ref:`LinearRunOnce<lnrunonce>` solver is a good choice for that level of the model.

So now the challenge is to select a good linear solver architecture for the :code:`cycle` group.
One possible approach is to use the :ref:`LinearBlockGS<linearblockgs>` solver for the :code:`cycle`,
and then assign additional solvers to the aerodynamics and structural analyses.

.. note::
    Choosing LinearBlockGaussSeidel is analogous to solving the nonlinear system with a NonLinearBlockGaussSeidel solver.

    Despite the analogy, it is not required nor even advised that your linear solver architecture match your nonlinear solver architecture.
    It could very well be a better choice to use the :ref:`PETScKrylov<petscKrylov>` solver for the :code:`cycle` level,
    even if the :ref:`NonlinearBlockGS<nlbgs>` solver was set as the nonlinear solver.

The :ref:`LinearBlockGS<linearblockgs>` solver requires that any implicit components underneath it have their own linear
solvers to converge their part of the overall linear system. So a :ref:`PETScKrylov<petsckrylov>` solver is used for :code:`aero`
and a :ref:`DirectSolver <directsolver>` is use for :code:`struct`. Looking back at the figure above, notice that these solvers
are all called out in their respective hierarchical locations.






