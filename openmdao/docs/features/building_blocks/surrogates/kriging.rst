.. _kriging:

***************************************
KrigingSurrogate, FloatKrigingSurrogate
***************************************



The `FloatKriginSurrogate` implements the same surrogate model as `KrigingSurrogate`, but it only predicts the mean of the approximated output. Note
that the `KrigingSurrogate` can also be executed in this mode by setting the `eval_rmse` option to False, which is the default.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_float_kriging
    :layout: code, output