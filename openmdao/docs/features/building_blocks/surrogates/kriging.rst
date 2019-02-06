.. _kriging:

****************
KrigingSurrogate
****************


The KrigingSurrogate implements a simple Kriging interpolation method based on Gaussian Processes
for Machine Learning (GPML) by Rasmussen and Williams. In the default configuration, the surrogate
computes the mean of the predicted value. KrigingSurrogate also has an option "eval_rmse", which can
be set to True to also compute the RMSE (root mean squared error).

Here is a simple example where a Kriging model is used to approximate the output a sinusoidal component.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_kriging
    :layout: code, output

Note: `FloatKriginSurrogate` implements the same surrogate model as `KrigingSurrogate` with `eval_rmse` set to False,
so that it only predicts the mean of the approximated output. This is deprecated and provided for backwards compatibility.


KrigingSurrogate Options
------------------------

All options can be passed in as arguments or set later by accessing the `options` dictionary.

.. embed-options::
    openmdao.surrogate_models.kriging
    KrigingSurrogate
    options


KrigingSurrogate Option Examples
--------------------------------

**eval_rmse**

By default, KrigingSurrogate only returns the mean of the predicted outputs. You can compute both the mean and the root
mean square prediction error by setting the "eval_rmse" option to True.  The most recent calculation of error is stored in
the component's metadata, and accessed as follows.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_kriging_options_eval_rmse
    :layout: code, output