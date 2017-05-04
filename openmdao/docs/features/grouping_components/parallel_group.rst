
Parallel Groups
---------------

When systems are added to a ParallelGroup, they will be executed in parallel, assuming that the ParallelGroup is
given an MPI communicator of sufficient size.  Adding subsystems to a ParallelGroup is no different than adding them
to a normal Group.  For example:


.. embed-code::
  openmdao.test_suite.groups.parallel_groups.initialize_variables.FanInGrouped


In this example, components *c1* and *c2* will be executed in parallel.



