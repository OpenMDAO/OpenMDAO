
*****************
pyOptSparseDriver
*****************

pyOptSparseDriver wraps the optimizer package pyOptSparse, which provides a common interface for 11 optimizers, some of which
are included in the package (e.g., SLSQP and NSGA2), and some of which are commercial products that must be obtained from their
respective authors (e.g. SNOPT). The pyOptSparse package is based on pyOpt, but adds support for sparse specification of
constraint Jacobians. Most of the sparsity features are only applicable when using the SNOPT optimizer.

.. note::
    The pyOptSparse package does not come included with the OpenMDAO installation. It is a separate optional package that can be obtained
    from  mdolab_.

In this simple example, we use the SLSQP optimizer to minimize the objective of SellarDerivativesGrouped.

.. embed-code::
    openmdao.drivers.tests.test_pyoptsparse_driver.TestPyoptSparseFeature.test_basic
    :layout: interleave

Optimizer Settings
==================

pyOptSparseDriver Options
-------------------------

.. embed-options::
    openmdao.drivers.pyoptsparse_driver
    pyOptSparseDriver
    options


The optimizers have a small number of unified options that can be controlled using the "options" dictionary. We have already shown how
to set the optimizer name. Next, the `print_results` option can be used to turn on or off the echoing of the pyOptSparse results when
the optimization finishes. The default is True, but here, we turn it off.

.. embed-code::
    openmdao.drivers.tests.test_pyoptsparse_driver.TestPyoptSparseFeature.test_settings_print
    :layout: interleave


Every optimizer also has its own specialized settings that allow you to fine-tune the algorithm that it uses. You can access these within
the `opt_setting` dictionary. These options are different for each optimizer, so to find out what they are, you need to read your
optimizer's documentation. We present a few common ones here.


SLSQP-Specific Settings
-----------------------

Here, we set a convergence tolerance for SLSQP:

.. embed-code::
    openmdao.drivers.tests.test_pyoptsparse_driver.TestPyoptSparseFeature.test_slsqp_atol
    :layout: interleave

Similarly, we can set an iteration limit. Here, we set it to just a few iterations, and don't quite reach the optimum.

.. embed-code::
    openmdao.drivers.tests.test_pyoptsparse_driver.TestPyoptSparseFeature.test_slsqp_maxit
    :layout: interleave


SNOPT-Specific Settings
-----------------------

SNOPT has many customizable settings. Here we show two common ones.

Setting the convergence tolerance:

.. embed-code::
    openmdao.drivers.tests.test_pyoptsparse_driver.TestPyoptSparseSnoptFeature.test_snopt_atol
    :layout: interleave

Setting a limit on the number of major iterations. Here, we set it to just a few iterations, and don't quite reach the optimum.

.. embed-code::
    openmdao.drivers.tests.test_pyoptsparse_driver.TestPyoptSparseSnoptFeature.test_snopt_maxit
    :layout: interleave

You can learn more about the available options in the SNOPT_Manual_.


.. toctree::
    :maxdepth: 1

.. _mdolab: https://github.com/mdolab/pyoptsparse

.. _SNOPT_Manual: http://www.sbsi-sol-optimize.com/manuals/SNOPT%20Manual.pdf
