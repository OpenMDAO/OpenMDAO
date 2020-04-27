.. _distributed_components:

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

We've already seen that by using :ref:`src_indices <connect_src_indices>`
we can connect an input to only a subset of an output variable.  
By giving different values for *src_indices* in each MPI process, we can
distribute computations on a distributed output across the processes.  

You tell the framework that a Component is a distributed component by setting its
:code:`distributed` option to True:


Component Options
-----------------

.. embed-options::
    openmdao.core.component
    Component
    options

.. note::

	If a Component is distributed then *all* of its outputs are distributed.

.. note::

	You should always specify *src_indices* when adding an input to a distributed 
	component unless your input is equal to the size of the entire distributed output.
	Otherwise, the assumed *src_indices* will be from 0 to 1 less than the 
	size of your input, which is probably not what you want.


Distributed Component Example
-----------------------------

The following example shows how to create a distributed component, `DistribComp`, 
that distributes its computation evenly across the available processes. A second 
component, `Summer`, sums the values from the distributed component into a scalar 
output value.  Note that a component that takes a distributed output as input does
not need to do anything special as OpenMDAO performs the required MPI operations 
to make the full value available.

These components can found in the OpenMDAO test suite:

.. embed-code::
  openmdao.test_suite.components.distributed_components.DistribComp

.. embed-code::
  openmdao.test_suite.components.distributed_components.Summer

This example is run with 2 processes and a size of 15:

.. embed-code::
  openmdao.core.tests.test_distribcomp.MPIFeatureTests.test_distribcomp_feature
  :layout: interleave
