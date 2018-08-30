.. _feature_parallel_fd:

************************************************************************************************
Speeding up Derivative Approximations with Parallel Finite Difference and Parrallel Complex Step
************************************************************************************************

If you have multiple processors available to you, it's possible to speed up the calculation of
approximated partials or totals by computing multiple columns of the approximated Jacobian matrix
simultaneously across multiple processes.  Setting up *parallel* finite difference or *parallel*
complex step is identical to setting up for serial execution, except for the additon of a single line
of code that sets the *num_par_fd* option on the System(s) where you want to compute the approximate
derivatives.  For example,

.. code-block:: python

    mysys.options['num_par_fd'] = 10  # compute 10 jacobian columns at a time in parallel


The *num_par_fd* option specifies the number of approximated jacobian columns that will be
computed in parallel, assuming that enough processes are provided when the problem script is
executed.  As a reminder, to run our script and give it 10 processes, we would do the following:

.. code-block:: console

  mpirun -n 10 python my_problem.py


More details about setting up for partial derivative approximation can be found
:ref:`here <feature_declare_partials_approx>` and an explanation of total derivative approximation
can be found :ref:`here <feature_declare_totals_approx>`.



