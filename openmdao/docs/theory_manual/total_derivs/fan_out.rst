.. _theory_fan_out: 

*******************************************
Parallelizing Fan-Out or Multipoint Models
*******************************************

A fan-out structure in a model is when you have a data path through the model that reaches a fans out to multiple components that can be run in parallel. 
For example, consider a toy optimization problem defined as follows: 

.. math:: 
    
    \begin{eqnarray}
    min & f \\
    wrt & x  \\
    st  & z > 0 .  
    \end{eqnarray}

With an associated model constructed using a
    
.. figure:: dependent_model.png
   :align: center
   :alt: A very simple model with a fan-out structure. 

   A very simple model with a fan-out structure

In this model, the components *Indep1* and *Comp1* are inexpensive components that compute data needed by the expensive *Con1* and *Con2* components. 
Since *Con1* and *Con2* have no dependence on each other they can be run in parallel, so we put them into a :ref:`parallel group<feature_parallel_group>`. 
There is a serial calculation in this model, for *Indep1* and *Comp1*, but it is very inexpensive compared to the constraint components in the parallel group. 
The objective function is also a serial bottle neck, but is also extremely inexpensive. 
So we can expect to get reasonable parallel scaling when running an analysis on a model set up like this. 

Another way to look at the structure of this model is to examine the partial derivative Jacobian structure. 
You get some dense rows from the upstream serial components that affect all the constraint components, 
but the rows associated with the constraints themselves display a block-diagonal (or in this case purely diagonal)
structure. 
The block-diaganol structure is exactly what you would expect in order to be able to run the constraint analyses in parallel. 

The fan-out model structure is common in engineering problems, particularly in multi-point problems where you want to evaluate the performance of a given system at multiple operating conditions during a single analysis. 

