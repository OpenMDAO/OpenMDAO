.. _feature_parallel_fd:

************************************************************************************************
Speeding up Derivative Approximations with Parallel Finite Difference and Parrallel Complex Step
************************************************************************************************

If you have multiple processors available to you, it's possible to speed up the calculation of
approximated partials or totals by computing multiple columns of the approximated Jacobian matrix
simultaneously across multiple processes.  Setting up *parallel* finite difference or *parallel*
complex step is identical to setting up for serial execution, except for the additon of a single
__init__ arg that sets *num_par_fd* on the System(s) where you want to compute the approximate
derivatives. The *num_par_fd* arg specifies the number of approximated jacobian columns that will be
computed in parallel, assuming that enough processes are provided when the problem script is
executed.


As a simple example, let's use parallel finite difference to compute the partial and total
derivatives using a simple model containing a component that computes its outputs by multiplying
its inputs by a constant matrix.  The partial jacobian for that component will be identical to
that constant matrix.  The component we'll use is shown below:

.. embed-code::
  openmdao.test_suite.components.matmultcomp.MatMultComp
  :layout: code

Our model will also contain an IndepVarComp that we'll connect to our MatMultComp, and we'll
compute total derivatives of our MatMultComp outputs with respect to our IndepVarComp output.


More details about setting up for partial derivative approximation can be found
:ref:`here <feature_declare_partials_approx>` and an explanation of total derivative approximation
can be found :ref:`here <feature_declare_totals_approx>`.


--------------------
Allocating processes
--------------------

In both cases we'll be running our python script under MPI using 3 processes.  We need 3 processes
because our model requires 1 process and we'll be using a *num_par_fd* value of 3, so the number
of processes we need is `3 * 1 = 3`.  In general, When we set *num_par_fd*, it acts as a
multiplier on the number of processes needed for any given system when running under MPI.
For example, if a given system requires N processes, then that same system when using parallel
finite difference will require `N * num_par_fd` processes.
This is because we're duplicating the given system `num_par_fd` times and using those duplicate
systems to solve for different columns of our approximated jacobian in parallel.

As a reminder, to run our script and give it 3 processes, we would do the following:

.. code-block:: console

  mpirun -n 3 python my_problem.py



--------
Examples
--------

First, let's compute the partial derivatives across our MatMultComp.  We'll use a matrix
with 6 columns and a *num_par_fd* value on our MatMultComp component of 3, meaning that each
of our 3 processes should compute 2 columns of the partial jacobian.

.. embed-code::
  openmdao.core.tests.test_parallel_fd.ParFDFeatureTestCase.test_fd_partials
  :layout: interleave


Next, let's compute the total derivatives of our MatMulComp outputs with respect to our
IndepVarComp output.  Again, we'll be using a *num_par_fd* value of 3 and a matrix having
6 columns, so each process should compute 2 columns of the total jacobian.  This time, however,
we set the *num_par_fd* on our model instead of on our MatMultComp.

.. embed-code::
  openmdao.core.tests.test_parallel_fd.ParFDFeatureTestCase.test_fd_totals
  :layout: interleave
