**********************
Distributed Components
**********************

At times when you need to perform a computation using large input arrays, you may
want to perform that computation in multiple processes, where each process
operates on some subset of the input values. This may be done purely for
performance reasons, or it may be necessary because the entire input will not fit
in the memory of a single machine.  In any case, this can be accomplished in
OpenMDAO using a distributed component.  A distributed component is a component
that operates on distributed variables. A variable is distributed if each process
contains only a part of the whole variable.

We've already seen that by using *src_indices* we can connect an input to only a
subset of an output variable.  By giving different values for *src_indices*
in each MPI process, we can distribute computations on a distributed output
across the processes.  You should always specify *src_indices* when connecting
to a distributed output unless your input is equal to the size of the entire
distributed output.  Otherwise, the assumed *src_indices* will be from 0 to
1 less than the size of your input, which is probably not what you want.

If a Component has *any* distributed outputs then *all* of its outputs are distributed.
You tell the framework that a Component is a distributed component by setting its
:code:`distributed` option to True:


Component Options
-----------------

.. embed-options::
    openmdao.core.component
    Component
    options

Distributed Component Example
-----------------------------

The following simple example shows how to create a distributed component that
distributes its computation evenly across the available processes, up to a
maximum of 5 processors.  It performs one computation on process 0 and a
different one on the other processors.  In this case we are only using 2
processes.


.. embed-code::
  openmdao.core.tests.test_distribcomp.MPIFeatureTests.test_distribcomp_feature
  :layout: interleave
