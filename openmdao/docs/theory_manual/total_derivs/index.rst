.. _theory_total_derivatives: 

*****************************************
How OpenMDAO Calculates Derivatives
*****************************************

This is a comprehensive document about how OpenMDAO solves for total derivatives. 
It is designed to be read in order, with later sections assuming understanding of prior ones. 

Terminology
---------------
Before diving into how OpenMDAO solves for total derivatives, it is important that we define a pair of key terms. 
Within the context of an OpenMDAO model we recognize two types of derivatives: 

    * Partial Derivative: Derivatives of the outputs of a single component with respect to that components inputs
    * Total Derivative: Derivative of an objective or constraint with respect to design variables

Partial derivatives are either :ref:`provided by the user<feature_specify_partials>`, or they can be :ref:`computed numerically using finite-difference or complex-step<feature_declare_partials_approx>`. 
Although they are an important subject, this document is focused on the computation of total derivatives which OpenMDAO computes by solving a linear system. 
Since we are focused on total derivatives here, we will assume that the partial derivatives are known a priori. 


Unified Derivatives Equations
-------------------------------

Total derivative calculations in OpenMDAO are based on the `Unified Derivative Equations`_
These equations allow you to compute total derivatives across any system of nonlinear equations, by solving a linear system in the forward (direct) form: 

.. math:: 
    
    \left[\frac{\partial \mathcal{R}}{\partial U}\right] \left[\frac{du}{dr}\right] = \left[ I \right],

or by solving a linear system in the reverse (adjoint) form: 

.. math:: 

    \left[\frac{\partial \mathcal{R}}{\partial U}\right]^T \left[\frac{du}{dr}\right]^T = \left[ I \right].

The matrix :math:`\left[\frac{\partial \mathcal{R}}{\partial U}\right]` is composed of all the partial derivatives from the model components, and the solution to the linear system gives the total derivatives :math:`\left[\frac{du}{dr}\right]`. 
So total derivative computations in OpenMDAO effectively amount to solving this linear system in either forward or reverse form. 
In forward form, one linear solve is required per design variable. 
In reverse form, one linear solve is required per objective and constraint. 
Hence, in many cases, selecting between forward and reverse is just a matter of counting how many design variables, and constraints you have and picking whichever form yields the least number of linear solves. 


Although the two unified derivatives equations are very simple, solving them efficiently for a range of different kinds of models  requires careful implementation. 
There are a number of different things that you, as a user of OpenMDAO, can do to control how the linear solutions are performed that will have an impact on both the speed and accuracy of the linear solution. 


Setting Up a Model for Efficient Linear Solves
---------------------------------------------------

A deeper understanding of how OpenMDAO solves this linear system is useful in understanding when to apply certain features 
and may also help you structure your model to make the most effective use of them. 
The explanation of OpenMDAO's features for improving linear solver performance are broken up into three sections below:  

.. _Unified Derivative Equations: http://mdolab.engin.umich.edu/content/review-and-unification-discrete-methods-computing-derivatives-single-and-multi-disciplinary

.. toctree::
    :maxdepth: 1

    coupled_vs_uncoupled.rst
    assembled_vs_matrixfree.rst
    hierarchical_linear_solver.rst

Algorithms To Reduce the Cost of the Linear Solves for Special Cases
----------------------------------------------------------------------

There are certain cases where it is possible to further improve linear solver performance 
via the application of specialized algorithms. 
The usage of these algorithms is detailed here. 

.. toctree::
    :maxdepth: 1

    vectorized.rst
    fan_out.rst
    separable.rst


