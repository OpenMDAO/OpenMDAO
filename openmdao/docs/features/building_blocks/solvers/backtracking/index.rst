.. _feature_line_search:

***********************
Linesearch/Backtracking
***********************

Backtracking line searches are subsolvers that can be specified in the `line_search` attribute
of a NewtonSolver, and are used to pull back to a reasonable point when a Newton step goes too far. This
can occur when a step causes output variables to exceed their specified lower and upper bounds. It can
also happen in more complicated problems where a full Newton step happens to take you well past the nonlinear solution,
even to an area where the residual norm is worse than the initial point. Specifying a value for line_search can
help alleviate these problems and improve robustness of your Newton solve.

There are two different backtracking line-search algorithms in OpenMDAO:

.. toctree::
    :maxdepth: 1

    armijo_goldstein.rst
    bounds_enforce.rst
    relaxation.rst

