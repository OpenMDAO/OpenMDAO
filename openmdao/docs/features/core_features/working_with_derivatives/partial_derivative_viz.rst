.. _feature_check_partials_viz:

**********************************************************
Visually Checking Partial Derivatives with Matrix Diagrams
**********************************************************


The function :code:`partial_deriv_plot` lets you see a visual representation of the values
returned by :code:`check_partials`.

.. autofunction:: openmdao.utils.visualization.partial_deriv_plot
    :noindex:

Here are two examples of its use. Note that in these examples, the :code:`compute_partials` method intentionally
computes the incorrect value so that the plots show how this function can be used to detect such errors.

With the default value :code:`binary` equal to :code:`True`, the plots
will only show the presence of a non-zero derivative, not the value.

.. embed-code::
    openmdao.utils.tests.test_visualization.TestFeatureVisualization.test_partial_deriv_plot
    :layout: code, plot
    :scale: 90
    :align: center

----

With the value :code:`binary` equal to :code:`False`, the plots show the actual value.

.. embed-code::
    openmdao.utils.tests.test_visualization.TestFeatureVisualization.test_partial_deriv_non_binary_plot
    :layout: code, plot
    :scale: 90
    :align: center

.. tags:: Derivatives