.. _feature_assembled_jacobian:

*******************************
In-Memory Assembly of Jacobians
*******************************

When you have groups, or entire models, that are small enough to fit the entire Jacobian into memory,
you can have OpenMDAO actually assemble the partial-derivative Jacobian in memory.
In many cases this can yield a substantial speed up over the default,
:ref:`matrix-free<theory_assembled_vs_matrix_free>` implementation in OpenMDAO.

.. note::
    Assembled Jacobians are especially effective when you have a deeply-nested hierarchy with a
    large number of components and/or variables. See the
    :ref:`Theory Manual entry on assembled Jacobians<theory_assembled_vs_matrix_free>` for more
    details on how to best select which type of Jacobian to use.


To use an assembled Jacobian, you set the :code:`assemble_jac` option of the linear solver that
will use it to True.  The type of the assembled jacobian will be determined by the value of
:code:`options['assembled_jac_type']` in the solver's containing system.
There are two options of 'assembled_jac_type' to choose from, `dense` and `csc`.

.. note::
    `csc` is an abbreviation for `compressed sparse column`. `csc` is one of many sparse storage schemes that
    allocate contiguous storage in memory for the nonzero elements of the matrix, and perhaps a limited number of zeros.
    For more information, see
    `Compressed sparse column <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)>`_.


For example:

.. code-block:: python

    model.options['assembled_jac_type'] = 'dense'
    model.linear_solver = DirectSolver(assemble_jac=True)


'csc' is the default, and you should try that first if you're not sure of which one to use.
Most problems, even if they have dense sub-Jacobians from each component, are fairly sparse at
the model level and the
:ref:`DirectSolver<directsolver>` will usually be much faster with a sparse factorization.

.. note::

   You are allowed to use multiple assembled Jacobians at multiple different levels of your model hierarchy.
   This may be useful if you have nested nonlinear solvers to converge very difficult problems.

-------------
Usage Example
-------------

In the following example, borrowed from the :ref:`newton solver tutorial<defining_icomps_tutorial>`,
we assemble the Jacobian at the same level of the model hierarchy as the :ref:`NewtonSolver<nlnewton>`
and :ref:`DirectSolver<directsolver>`. In general, you will always do the assembly at the same level
as the linear solver that will make use of the Jacobian matrix.

.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_plain_newton_assembled
    :layout: code, output
