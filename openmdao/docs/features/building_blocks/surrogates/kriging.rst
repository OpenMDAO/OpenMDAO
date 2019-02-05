.. _kriging:

****************
KrigingSurrogate
****************


The KrigingSurrogate implements a simple Kriging interpolation method based on Gaussian Processes
for Machine Learning (GPML) by Rasmussen and Williams. In the default configuration, the surrogate
outputs the mean of the predicted value. KrigingSurrogate also has an option "eval_rmse", which can
be set to True to output a tuple of mean and RMSE.

Here is a simple example where a Kriging model is used to approximate the output a sinusoidal component.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_kriging
    :layout: code, output

KrigingSurrogate Options
------------------------

.. embed-options::
    openmdao.surrogate_models.kriging
    KrigingSurrogate
    options

Note: `FloatKriginSurrogate` implements the same surrogate model as `KrigingSurrogate` with `eval_rmse` set to False,
so that it only predicts the mean of the approximated output. This is deprecated and provided for backwards compatibility.


KrigingSurrogate Option Examples
--------------------------------

**bound_enforcement**