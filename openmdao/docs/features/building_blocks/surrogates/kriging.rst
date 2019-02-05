.. _kriging:

****************
KrigingSurrogate
****************



Note: the `FloatKriginSurrogate` implements the same surrogate model as `KrigingSurrogate` with `eval_rmse` set to False,
so that it only predicts the mean of the approximated output. This is deprecated and provided for backwards compatibility.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_kriging
    :layout: code, output

KrigingSurrogate Options
------------------------

.. embed-options::
    openmdao.surrogate_models.kriging
    KrigingSurrogate
    options