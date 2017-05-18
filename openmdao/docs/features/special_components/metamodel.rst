.. index:: MetaModel Example

MetaModel Component
---------------------------

`MetaModel` lets you quickly create a component with surrogate models
used to compute the outputs based on training data. Generally, this is
used to construct a low computational cost replacement for computationally
expensive components.

You can define a `MetaModel` with as many inputs and outputs as you like,
and you can also use a different surrogate model for each output.

.. note::

    What's the difference between a `MetaModel` and a surrogate model? In
    OpenMDAO, "surrogate model" refers to the model for a single response, and
    `MetaModel` represents a collection of surrogate models trained at the
    same locations in the design space.

The following example demonstrates a simple `Problem` in which a
`MetaModel` component uses surrogates to mimic the sine and cosine functions.

In this example, the `MetaModel` component ``trig`` has a single input,
``x``, and two outputs, ``sin_x`` and ``cos_x``.

A `FloatKrigingSurrogate` is given as the surrogate for the ``sin_x`` output.
Although no surrogate has been given for the ``cos_x`` output, a
``default_surrogate`` is specified for the component. Any output which has
not had a surrogate assigned will use one of the default type.
If ``default_surrogate`` is not specified, then a surrogate must be
given for all outputs.

Training data is provided as metadata to the ``trig`` component using the variable
names prefixed with ``train:``.  This can be done anytime before the `MetaModel`
runs for the first time.

The first time a `MetaModel` runs, it will train the surrogates using the
training data that has been provided and then it will predict the output
values. The training step only occurs on the first run.

.. embed-test::
    openmdao.components.tests.test_meta_model.MetaModelTestCase.test_metamodel_feature

.. tags:: MetaModel, Examples
