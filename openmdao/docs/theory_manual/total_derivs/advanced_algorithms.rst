Advanced Linear Solver Algorithms for Special Cases
---------------------------------------------------

There are certain cases where it is possible to further improve linear solver performance via the application of specialized algorithms.
In some cases, the application of these algorithms can have an impact on whether you choose the forward or reverse mode for derivative solves.
This section details the types of structures within a model that are necessary in order to benefit from these algorithms.

.. toctree::
    :maxdepth: 1

    separable.rst
    fan_out.rst
    vectorized.rst