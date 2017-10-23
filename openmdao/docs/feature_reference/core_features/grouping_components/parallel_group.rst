***************
Parallel Groups
***************

When systems are added to a ParallelGroup, they will be executed in parallel, assuming that the ParallelGroup is
given an MPI communicator of sufficient size.  Adding subsystems to a ParallelGroup is no different than adding them
to a normal Group.  For example:


.. embed-test::
  openmdao.core.tests.test_parallel_groups.TestParallelGroups.test_fan_in_grouped_feature


In this example, components *c1* and *c2* will be executed in parallel, provided that the ParallelGroup is given 2
MPI processes.  If the name of the python file containing our example were `my_par_model.py`, we could run it under
MPI and give it 2 processes using the following command:


.. code-block:: console

  mpirun -n 2 python my_par_model.py


.. note::

  This will only work if you've installed the mpi4py and petsc4py python packages, which are not installed by default
  in OpenMDAO.
