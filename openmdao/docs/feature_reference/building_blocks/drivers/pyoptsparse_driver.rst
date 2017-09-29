
******************
pyoptsparse Driver
******************

The pyOptSparseDriver wraps the optimizer package pyOptSparse, which provides a common interface for 11 optimizers, some of which
are included in the package (e.g., SLSQP and NSGA2), and some of which are commercial products that must be obatined from their
respective authors (e.g. SNOPT). The pyOptSparse package is based off of pyOpt, but it adds support for sparse specification of
constraint jacobians. Most of the sparsity features are only applicable when using the SNOPT optimizer.

.. toctree::
    :maxdepth: 1

