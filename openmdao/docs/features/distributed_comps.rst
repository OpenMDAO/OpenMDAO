
Distributed Components
----------------------

At times when you need to perform a computation using large input arrays, you may
want to perform that computation in multiple processes, where each process
operates on some subset of the input values. This may be done purely for
performance reasons, or it may be necessary because the entire input will not fit
in the memory of a single machine.  In any case, this can be accomplished in
OpenMDAO using a distributed component.

We designate that a component is distributed by setting its *distributed*
attribute to `True`.  That tells the framework that all of its outputs should
be considered as distributed instead of duplicated. If a variable is
distributed, that means that the each process contains only a part of the whole
variable.  If a variable is duplicated, then there will be an identical copy
of that variable in every process.

We've already seen that by using *src_indices* we can connect an input to only a
subset of an output variable.  By giving different values for *src_indices*
in each MPI process, we can distribute computations on a distributed output
across the processes.  If we connect an input to a distributed output without
specifying *src_indices*, then the entire distributed output will be copied
into the input during a transfer.  Note that this means that the input must
be large enough to contain the entire distributed output.

The following simple example shows how to create a distributed component that
distributes its computation evenly across the available processes, up to a
maximum of 5 processors.  It performs one computation on process 0 and a
different one on the other processors.  In this case we are only using 2
processes.


.. embed-test::
  openmdao.core.tests.test_distribcomp.MPITests.test_distribcomp_feature
