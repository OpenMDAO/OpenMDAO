.. _feature_MetaModelUnStructuredComp:

*************************
MetaModelUnStructuredComp
*************************

`MetaModelUnStructuredComp` lets you quickly create a component with surrogate models
used to compute the outputs based on unstructured training data. Generally, this is
used to construct a low-computational-cost replacement for computationally
expensive components.

You can define `MetaModelUnStructuredComp` with as many inputs and outputs as you like,
and you can also use a different surrogate model for each output.

.. note::

    What's the difference between `MetaModelUnStructuredComp` and a surrogate model? In
    OpenMDAO, "surrogate model" refers to the model for a single response, and
    `MetaModelUnStructuredComp` represents a collection of surrogate models trained at the
    same locations in the design space.

MetaModelUnStructuredComp Options
---------------------------------

.. embed-options::
    openmdao.components.meta_model_unstructured_comp
    MetaModelUnStructuredComp
    options

Simple Example
--------------

The following example demonstrates a simple `Problem` in which a
`MetaModelUnStructuredComp` uses surrogates to mimic the sine and cosine functions.

In this example, the `MetaModelUnStructuredComp` ``trig`` has a single input,
``x``, and two outputs, ``sin_x`` and ``cos_x``.

`FloatKrigingSurrogate` is given as the surrogate for the ``sin_x`` output.
Although no surrogate has been given for the ``cos_x`` output, a
``default_surrogate`` is specified for the component. Any output which has
not had a surrogate assigned will use one of the default type.
If ``default_surrogate`` is not specified, then a surrogate must be
given for all outputs.


The first time a `MetaModelUnStructuredComp` runs, it will train the surrogates using the
training data that has been provided, and then it will predict the output
values. This training step only occurs on the first run.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelTestCase.test_metamodel_feature
    :layout: code, output


Advanced usage
--------------

You can specify the training data after instantiation if you like, by setting the component's
:ref:`options<component_options>`. Training data is provided in the options to the ``trig``
component using the variable names prefixed with ``train:``.  This can be done anytime before
the `MetaModelUnStructuredComp` runs for the first time.

The inputs and outputs of a `MetaModelUnStructuredComp` are not limited to scalar values. The
following modified version of the example uses an array to predict sine and
cosine as a single output array of two values.  You will also note that the default
surrogate can be passed as an argument to the `MetaModelUnStructuredComp` constructor, as an
alternative to specifying it later.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelTestCase.test_metamodel_feature2d
    :layout: code, output

In addition, it's possible to vectorize the input and output variables so that you can
make multiple predictions for the inputs and outputs in a single execution of the
`MetaModelUnStructuredComp` component. This is done by setting the ``vec_size`` argument when
constructing the `MetaModelUnStructuredComp` component and giving it the number of predictions to make.  The following example vectorizes the ``trig``
component so that it makes three predictions at a time.  In this case, the input is
three independent values of ``x`` and the output is the corresponding predicted values
for the sine and cosine functions at those three points.  Note that a vectorized
`MetaModelUnStructuredComp` component requires the first dimension of all input and output variables
to be the same size as specified in the ``vec_size`` argument.


.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelTestCase.test_metamodel_feature_vector2d
    :layout: code, output


Using Surrogates That Not Define Linearize Method
-------------------------------------------------

In some cases, users might define surrogates but not define a `linearize` method. In this case, the
`MetaModelUnStructuredComp` derivatives will be computed using finite differences for the output variables that use that
surrogate. By default, the default values for the finite differencing method will be used.

If the user would like to specify finite differencing options, they can do so by calling the `declare_partials`
method in the component's `setup` or `configure` methods. This example, which uses a simplified surrogate with no
`linearize` method and no training, shows `declare_partials` called in `setup`.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelTestCase.test_feature_metamodel_use_fd_if_no_surrogate_linearize
    :layout: code, output


Complex step has not been tested with `MetaModelUnStructuredComp` and will result in an exception if used.


.. tags:: MetaModelUnStructuredComp, Component
