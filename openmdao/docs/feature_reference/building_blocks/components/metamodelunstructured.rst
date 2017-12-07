.. _feature_MetaModelUnStructured:

*******************************
MetaModelUnStructured Component
*******************************

`MetaModelUnStructured` lets you quickly create a component with surrogate models
used to compute the outputs based on unstructured training data. Generally, this is
used to construct a low computational cost replacement for computationally
expensive components.

You can define a `MetaModelUnStructured` with as many inputs and outputs as you like,
and you can also use a different surrogate model for each output.

.. note::

    What's the difference between a `MetaModelUnStructured` and a surrogate model? In
    OpenMDAO, "surrogate model" refers to the model for a single response, and
    `MetaModelUnStructured` represents a collection of surrogate models trained at the
    same locations in the design space.

The following example demonstrates a simple `Problem` in which a
`MetaModelUnStructured` component uses surrogates to mimic the sine and cosine functions.

In this example, the `MetaModelUnStructured` component ``trig`` has a single input,
``x``, and two outputs, ``sin_x`` and ``cos_x``.

A `FloatKrigingSurrogate` is given as the surrogate for the ``sin_x`` output.
Although no surrogate has been given for the ``cos_x`` output, a
``default_surrogate`` is specified for the component. Any output which has
not had a surrogate assigned will use one of the default type.
If ``default_surrogate`` is not specified, then a surrogate must be
given for all outputs.

Training data is provided as metadata to the ``trig`` component using the variable
names prefixed with ``train:``.  This can be done anytime before the `MetaModelUnStructured`
runs for the first time.

The first time a `MetaModelUnStructured` runs, it will train the surrogates using the
training data that has been provided and then it will predict the output
values. This training step only occurs on the first run.

.. embed-test::
    openmdao.components.tests.test_meta_model_unstructured.MetaModelTestCase.test_metamodel_feature
    :no-split:

The inputs and outputs of a `MetaModelUnStructured` are not limited to scalar values. The
following modified version of the example uses an array to predict sine and
cosine as a single output array of two values.  You will also note that the default
surrogate can been passed as an argument to the `MetaModelUnStructured` constructor, as an
alternative to specifying it later.

.. embed-test::
    openmdao.components.tests.test_meta_model_unstructured.MetaModelTestCase.test_metamodel_feature2d
    :no-split:

In addition, it's possible to vectorize the input and output variables so that you can
make multiple predictions for the inputs and outputs in a single execution of the
`MetaModelUnStructured` component. This is done by setting the ``vectorize`` argument when
constructing the `MetaModelUnStructured` component.  The following example vectorizes the ``trig``
component so that it makes three predictions at a time.  In this case, the input is
three independent values of ``x`` and the output is the corresponding predicted values
for the sine and cosine functions at those three points.  Note that a vectorized
`MetaModelUnStructured` component requires the first dimension of all input and output variables
to be the same size as specified in the ``vectorize`` argument.

.. embed-test::
    openmdao.components.tests.test_meta_model_unstructured.MetaModelTestCase.test_metamodel_feature_vector2d
    :no-split:

.. tags:: MetaModel, Examples
