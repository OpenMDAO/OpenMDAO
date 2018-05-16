.. _theory_vectorized_derivatives:

*************************************************************************
Vectorizing Derivative Solves for Array Variables At a Larger Memory Cost
*************************************************************************

When a model has a :ref:`feed-forward<theory_selecting_linear_solver>` (i.e. uncoupled) data path from design variables
to constraints, then you can use the default :ref:`LinearRunOnce<lnrunonce>` solver to compute total derivatives.
This solver will recurse down through your model, computing matrix-vector products as it goes, and will compute derivatives
one design variable at a time, or one constraint at a time in forward or reverse modes, respectively.
Consider what this means if you have a design variable or constraint composed of a large array.
In this case, the recursive algorithm is called once per entry in the variable.

While this recursive algorithm will compute the correct derivatives, if you have a deeply-nested model (i.e. many layers
of nested groups with lots of components) the overhead of that recursion can become significant. If you have a very large
array variable combined with a deeply-nested model, you are then calling the somewhat-expensive, recursive function many times.

In order to speed this up somewhat, OpenMDAO will allow you to vectorize the linear solves used to compute total derivatives.
Normally we perform a solve on a single right-hand-side vector, but when vectorized, OpenMDAO will allocate space for
multiple right-hand sides and multiple solution vectors, and then will recurse through the model only a single time to
compute all the derivatives for every entry in a given array variable. The advantage of this approach is that the number
of recursive traversals of the model is dramatically reduced, giving reduced overhead for large models.

While reduced computational overhead is advantageous, this approach also has the trade-off of an increased memory footprint.
As we noted, in order to use this algorithm, multiple right-hand-side vectors and solutions vectors must be allocated at the same time.

.. tip::

    If you are considering using vectorized total derivatives solves because you have a set of vectorized components in your model, you might also consider checking if your problem has a separable structure that can be taken advantage of with :ref:`simultaneous derivatives<theory_separable_variables>`.
    If you can use simultaneous derivatives, you get all of the computational speed-up, with none of the memory cost.
    If your problem isn't separable, then consider vectorized derivatives instead.

----------------------------
How Much Memory Does it Use?
----------------------------

OpenMDAO allocates vectors for the linear solves according to how many output variables there are and how large each one is.
How much additional memory is needed will depend on the details of your model.
OpenMDAO only needs to allocate enough memory for the vectorized solve of one variable at a time, so the memory cost can
be considered on a per-variable basis and will be governed by whichever variable has the largest size.

Consider a notional model (or a part of a model) with five design variables and five constraints, each one of size 1000.
This notional model is very simple, comprised of just two components: one :ref:`IndepVarComp<comp-type-1-indepvarcomp>` and one :ref:`ExplicitComponent<comp-type-2-explicitcomp>`.
So there are ten total outputs in the model, yielding a vector of size 10000 using 0.08 MB of memory.
OpenMDAO needs two vectors for each linear solve (one for the solution, and one for the right-hand side), using 0.16 MB for the pair.

A non-vectorized solve would require just the single pair of vectors,
but a vectorized derivative solve needs 1000 right-hand-side vectors and 1000 solution vectors all at once, using 160 MB of memory.

On modern computers, 160 MB of memory is not a significant amount, especially if it comes with the benefit of a large computational savings.
However, the above example only involved 2 components, and hence, the cost of the recursive solves won't be significant either.

Instead, consider what would happen if you had the same five design variables and five constraints of size 1000, but now
the model is comprised of fifteen different components connected in a feed-forward manner, arranged in some complex hierarchy.
Each component will have its own five output variables, giving a total of 75 variables of size 1000 each.
In this larger model, each vector is now of size 75000, using 0.6 MB of memory.
So a non-vectorized solve will use 1.2 MB of memory but the vectorized solve will now use 1200 MB (1.2 GB) of memory.

This illustrates how the memory cost of vectorization grows with two key factors:

    #. The number of output variables in a model
    #. The size of the output variable for which you want to vectorize the total derivative solve

Ultimately, you have to weigh the memory cost and compute savings to determine if this feature is good for your specific use case.

--------------------------------------------------
Usage With Components That Use Assembled Jacobians
--------------------------------------------------

If all components in your model are using the :ref:`compute_partials <comp-type-2-explicitcomp>` or :ref:`linearize <comp-type-3-implicitcomp>`
methods to provide OpenMDAO with their partial derivatives, then you do not need to do anything else in order to support vectorized derivatives solves.

Internally, OpenMDAO will switch from doing matrix-vector products to vectorized matrix-matrix products.
Essentially in either case, if you stored the partial-derivative Jacobian in a variable :code:`A` and your vector as variable :code:`b`, OpenMDAO will always do

.. code::

    c = A.dot(b)

This NumPy syntax works whether b is a vector or a collection of vectors that have been stacked together (vectorized),
which is why you don't need to do anything differently when using assembled Jacobians.

--------------------------------------------------------------
Usage With Components That Use Matrix-Free Partial Derivatives
--------------------------------------------------------------

If you have any components that use the matrix-free APIs,

    #. :ref:`compute_jacvec_product<comp-type-2-explicitcomp>`
    #. :ref:`apply_linear<comp-type-3-implicitcomp>`

Then you need to implement additional methods in order to use vectorized derivative solves.
The new methods are necessary because the linear operators themselves need to be vectorized and it's not possible for
OpenMDAO to efficiently do that for you.

    #. :ref:`compute_multi_jacvec_product<comp-type-2-explicitcomp>`
    #. :ref:`apply_multi_linear<comp-type-3-implicitcomp>`


.. warning::

    You only need to implement these additional API methods **IF** you plan to use these components with vectorized derivative solves!

