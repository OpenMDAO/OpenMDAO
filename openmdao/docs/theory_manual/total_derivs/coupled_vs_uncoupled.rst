****************************************************************
Determining What Kind of Linear Solver You Need
****************************************************************

Since total derivatives are computed by solving the unified derivatives equations, there is always some kind of linear solver used by the framework whenever :ref:`compute_totals<feature_compute_totals>` is called.
However, the specific type of linear solver that should be used will vary greatly depending on the underlying model structure.
The most basic distinguishing feature of a model that governs what kind of linear solver should be used is the presence of any coupling.

----------------------------
Uncoupled Models
----------------------------

If you have a completely uncoupled model, then the partial derivative Jacobian matrix will have an lower-triangular structure.
If you are using *reverse* mode, then the left hand side of the unified derivatives equations will be the transpose-Jacobian and will have an upper triangular structure.
The upper-triagular transpose jacobian structure is notable because it can also be seen in the :ref:`n2 diagram<om-command-view_model>`
that OpenMDAO can produce.

The resulting linear system can be solved using a block forward or backward substitution algorithm.
Alternatively you could view the solution algorithm as a single iteration of a block Gauss-Seidel algorithm.
In OpenMDAO, this solution algorithm is implemented via the :ref:`LinearRunOnce<lnrunonce>` solver.
This is the default solver used by OpenMDAO on all :ref:`Group<feature_grouping_components>`.


----------------------------
Coupled Models
----------------------------

Coupled models will always have a non-triangular structure to their partial derivative Jacobian, which means they will need always need more than the default linear solver setup as well.
There are two basic categories of linear solver that can be used in this case:

    #. direct solvers (e.g. :ref:`DirectSolver<directsolver>`)
    #. iterative solvers (e.g. :ref:`LinearBlockGS<linearblockgs>`, :ref:`ScipyKrylov<scipyiterativesolver>`)

Direct solvers make use of a the Jacobian matrix, assembled in memory, in order to compute an inverse or a factorization that can be used to solve the linear system.
Conversely, Iterative linear solvers find the solution to the linear system without ever needing to access the Jacobian matrix directly.
The search for solution vectors that drive the linear residual to 0 using only matrix-vector products.

The decision about which type of solver to use is heavily model dependent, and is discussed in a later section of the theory manual.
The key idea is that **some** kind of linear solver is needed when there is coupling in your model.


Analogy to the Non-Linear Solvers
------------------------------------

Any coupling in your model will affect both the linear and non-linear solves, and thus impact which type of linear and non-linear solvers you use.

In the most basic case, an uncoupled model will use the default :ref:`NonLinearRunOnce <nlrunonce>` and the :ref:`LinearRunOnce<lnrunonce>` solvers.
These *RunOnce* drivers are a special degenerate class of solver, which can't handle any kind of coupling or implicitness in a model.
More complex models, with coupling, will require and iterative nonlinear solver to converge the non-linear model.
Any model that requires an iterative non-linear solver you will also need a linear solver other than :ref:`LinearRunOnce<lnrunonce>` solvers.

-------------------------------------
Mixed Coupled-Uncoupled Models
-------------------------------------

In the broadest sense, if there is any coupling in a model then it could be classified as a coupled model.
At the top of the model hierarchy, you could assign an appropriate nonlinear and linear solver, :ref:`NewtonSolver<nlnewton>` and :ref:`DirectSolver<directsolver>` for example, and assuming that the the solvers converged you would get the correct answer.

However, OpenMDAO allows you to build up a hierarchical model structure using :ref:`groups<feature_grouping_components>`.
That same hierarchy can also be leveraged to develop a more sophisticated hierarchical solver structure to more efficiently and robustly solve the linear system.

Consider, as an example, the :ref:`Sellar Problem<sellar>` from the :ref:`Multidisciplinary Optimization User Guide <user_guide_multi_disciplinary_opt>`.
In that problem, coupling is created by a cyclic connection between the :code:`d1` and :code:`d2` components.
You can see that show up clearly in the n2 diagram below, because there are off-diagonal terms both above and below the diagonal inside the :code:`cycle` group.

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarMDALinearSolver

.. raw:: html
    :file: sellar_n2.html

Since there is coupling in this model, as we've already established, there must also be some linear solver there to deal with it.
One option would be to assign the :ref:`DirectSolver <directsolver>` right at the top level of the model, and have it compute an inverse of the full Jacobian.
While that would certainly work, you're taking an inverse of a larger matrix than you really need to.

Instead, as we've shown in the code above, you can assign the :ref:`DirectSolver <directsolver>` at the :code:`cycle` level instead.
The top level of the hierarchy will then have the default :ref:`LinearRunOnce<lnrunonce>` solver in it.
Effectively, the direct solver is being used to compute the coupled semi-total derivatives across the :code:`cycle` group, which then makes the model have a feed-forward data path that can be solved with forward of back substitution (depending on which solve mode you used).

To illustrate that visually, you can *right-click* on the cycle group in the n2 diagram above.
This will collapse the cycle group to a single box, and you will see the resulting uncoupled, upper-triangular matrix structure that results.
That is why you can think of this model as a mixed coupled-uncoupled model, because once you converge the coupling inside :code:`cycle` the rest of the model is uncoupled.

Practically speaking, for tiny problem like :ref:`Sellar<sellar>` there won't be any real performance difference between putting the :ref:`DirectSolver <directsolver>` at the top, vs down in the :code:`cycle` group.
However, in real models with hundreds or thousands of variables the effect can be much more pronounced if you're trying to invert a dense 10000x10000 matrix.
More importantly, if you have models with high-fidelity codes like CFD or FEA in the hierarchy,
you simply may not be able to use a :ref:`DirectSolver <directsolver>` at the top of the model, but there may still be a portion of the model where it makes sense.
So understanding how to take advantage of the model hierarchy in order to customize the linear solver behavior becomes more important as the model complexity increases.

