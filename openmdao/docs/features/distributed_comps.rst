
Distributed Components
----------------------

At times when you need to perform a computation using large input arrays, you may
want to perform that computation in multiple processes, where each process
operates on some subset of the input values. This may be done purely for
performance reasons, or it may be necessary because the entire input will not fit
in the memory of a single machine.  In any case, this can be accomplished in
OpenMDAO using a distributed component.

We've already seen that by using src_indices we can connect an input to only a
subset of an output variable.  By giving different values for *src_indices*
in each MPI process, we can distribute the computations across the processes.

The following simple example shows how to create a distributed component that
distriibutes its computation evenly across the available processes, up to a
maximum of 5 processors.  It performs one computation on process 0 and a
different one on the other processors.  In this case we are only using 2
processes.


.. embed-test::
  openmdao.core.tests.test_distribcomp.MPITests.test_distribcomp_feature
