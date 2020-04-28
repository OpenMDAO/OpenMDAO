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


Distributed Component Example
-----------------------------

The following example shows how to create a distributed component, `DistribComp`, 
that distributes its computation evenly across the available processes. A second 
component, `Summer`, sums the values from the distributed component into a scalar 
output value.  

These components can found in the OpenMDAO test suite:

.. embed-code::
    openmdao.test_suite.components.distributed_components.DistribComp

.. note::

    In this example component, we have explicitly specified *src_indices* when adding
    the input. This is not really necessary in this case, because it replicates the
    default behavior. If no *src_indices* are specified, OpenMDAO will assume an offset
    that is the sum of the sizes in all ranks up to the current rank and a range equal
    to the specified size (the size is given per the usual arguments to :code:`add_input`).

.. embed-code::
    openmdao.test_suite.components.distributed_components.Summer

.. note::

    A component that takes a distributed output as input does not need to do anything
    special as OpenMDAO performs the required MPI operations to make the full value 
    available.

This example is run with two processes and a :code:`size` of 15:

.. embed-code::
    openmdao.core.tests.test_distribcomp.MPIFeatureTests.test_distribcomp_feature
    :layout: interleave


Distributed Component with Derivatives
--------------------------------------

Derivatives can be computed for distributed components as shown in the following
variation on the example.  Also, in this version, we have taken advantage of the automatic
determination of *src_indices*.


.. embed-code::
    openmdao.test_suite.components.distributed_components.DistribCompDerivs

.. embed-code::
    openmdao.test_suite.components.distributed_components.SummerDerivs


This example is again run with two processes and a :code:`size` of 15.  We can use
:ref:`assert_check_partials<feature_unit_testing_partials>` to verify that
the partial derivatives are calculated correctly.

.. embed-code::
    openmdao.core.tests.test_distrib_derivs.MPIFeatureTests.test_distribcomp_derivs_feature
    :layout: interleave
