.. _sparse-totals:

****************************************
Computing Sparsity of Total Derivatives
****************************************


If your total derivative jacobian is sparse, certain optimizers, e.g., the SNOPT optimizer
in :ref:`pyOptSparseDriver<feature_pyoptsparse>`, can take advantage of this to improve the
performance of computing total derivatives.

Computing the sparsity of the jacobian can be performed either statically or dynamically.


Dynamic Determination of Sparsity
=================================

Sparsity can be computed at runtime, shortly after the driver begins the optimization.
This has the advantage of simplicity and robustness to changes in the model, but adds
the cost of the sparsity computation to the run time of the optimization.  For a typical
optimization, however, this cost will be small.  Activating dynamic sparsity detection
is simple.  Just set the `dynamic_derivs_sparsity` option on the driver.  For example:

.. code-block:: python

    prob.driver.options['dynamic_derivs_sparsity'] = True

If you want to change the number of compute_totals calls that the algorithm uses to
compute the jacobian sparsity (default is 3), you can set the `dynamic_derivs_repeats`
option. For example:

.. code-block:: python

    prob.driver.options['dynamic_derivs_repeats'] = 2


Whenever a dynamic sparsity is computed, the sparsity is written to a file called *sparsity.json*
for later inspection.


Static Determination of Sparsity
================================

To get rid of the runtime cost of computing the coloring, you can precompute it using the
:code:`openmdao sparsity` command line tool.  The sparsity dictionary will be displayed on
the terminal and can be cut-and-pasted into your python script, or you can specify an output
file on the command line and the sparsity dictionary will be written in JSON format to the
specified file.  Here's an example of writing the sparsity information to the terminal:

.. embed-shell-cmd::
    :cmd: openmdao sparsity circle_opt.py
    :dir: ../test_suite/scripts


Once the sparsity data exists, you tell OpenMDAO about it by calling the `set_total_jac_sparsity`
method on the driver.  For example:


.. code-block:: python

    sparsity = {
        "circle.area": {
           "indeps.x": [[], [], [1, 10]],
           "indeps.y": [[], [], [1, 10]],
           "indeps.r": [[0], [0], [1, 1]]
        },
        "r_con.g": {
           "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
           "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
           "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
        },
        "theta_con.g": {
           "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
           "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
           "indeps.r": [[], [], [5, 1]]
        },
        "delta_theta_con.g": {
           "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
           "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
           "indeps.r": [[], [], [5, 1]]
        },
        "l_conx.g": {
           "indeps.x": [[0], [0], [1, 10]],
           "indeps.y": [[], [], [1, 10]],
           "indeps.r": [[], [], [1, 1]]
        }
    }

    # we would specify total jacobian sparsity by calling this on our driver
    prob.driver.set_total_jac_sparsity(sparsity)


If we want to write the sparsity output to a JSON file instead, the command would look like this:

.. code-block:: none

    openmdao sparsity circle_opt.py -o sparsity.json


and we would specify the sparsity in our python script as follows:

.. code-block:: python

    # we would specify total jacobian sparsity by calling this on our driver
    prob.driver.set_total_jac_sparsity('sparsity.json')

.. note::

  The above code assumes that we're running our script in the same directory where we put the json file.
