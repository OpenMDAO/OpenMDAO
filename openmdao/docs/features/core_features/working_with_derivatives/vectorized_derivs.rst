.. _feature_vectorized_derivatives: 

#################################################
Vectorizing Linear Solves for Feed Forward Models
#################################################

I you have an optimization constraint composed of a large array, or similarly a large array design variable, then there will be one linear solve for each entry of that array. 
It is possible to speed up the derivative computation by vectorizing the linear solve associated with the design variable or constraint, 
though the speed up comes at the cost of some additional memory allocation within OpenMDAO. 
 
.. note:: 
    
    Vectorizing derivatives is only viable for variables/constraints that have a purely feed-forward data path through the model. 
    If there are any solvers in the path between your variable and the objective/constraint of your model then you should not use this feature!
    See the :ref:`theory manual on vectorized derivatives<theory_vectorized_derivaties>` for a detailed explanation of how this feature works. 


-------------
Usage Example
-------------