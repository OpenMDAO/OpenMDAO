.. _component_options:

*******************************************
Component Options (Arguments to Components)
*******************************************

The primary jobs of a component, whether explicit or implicit, are to define inputs and outputs
and to do the mapping that computes the outputs given the inputs.
Often, however, there are incidental parameters that affect the behavior of the component,
but which are not considered input variables in the sense of being computed as an output of another component.

OpenMDAO provides a way of declaring these parameters, which are contained in an
`OptionsDictionary` named *options* that is available in every component. Options
associated with a particular component must be declared in the `initialize` method
of the component definition. A default value can be provided as well as various checks
for validity, such as a list of acceptable values or types.

The full list of options is shown in the method signature below.

.. automethod:: openmdao.utils.options_dictionary.OptionsDictionary.declare
    :noindex:

Option values are typically passed at component instantiation time as keyword arguments,
which are automatically assigned into the option dictionary. The options are then available
for use in the component's other methods, such as `setup` and `compute`.

Alternatively, values can be set at a later time, in another method of the component
(except for `initialize`) or outside of the component definition after the component is
instantiated.

A Simple Example
----------------

Options are commonly used to specify the shape or size of the component's input and output
variables, such as in this simple example.

.. embed-code::
    openmdao.test_suite.components.options_feature_vector

.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple
    :layout: interleave


Not setting a default value when declaring an option implies that the value must be set by the user.

In this example, 'size' is required; We would have gotten an error if we:

1. Did not pass in 'size' when instantiating *VectorDoublingComp* and
2. Did not set its value in the code for *VectorDoublingComp*.


.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple_fail
    :layout: interleave


Metadata Types
--------------

Options are not limited to simple types like :code:`int`.  In the following example, the
component takes a `Numpy` array as an option:


.. embed-code::
    openmdao.test_suite.components.options_feature_array

.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple_array
    :layout: interleave


It is even possible to provide a function as an option:


.. embed-code::
    openmdao.test_suite.components.options_feature_function

.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple_function
    :layout: interleave

Providing Default Values
------------------------

One reason why using options is convenient is that a default value can be specified,
making it optional to pass the value in during component instantiation.

.. embed-code::
    openmdao.test_suite.components.options_feature_lincomb

.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_with_default
    :layout: interleave

In this example, both 'a' and 'b' are optional, so it is valid to pass in 'a', but not 'b'.

Specifying Values or Types
--------------------------

Another commonly-used feature of options is specifying acceptable values or types.
If only the list of acceptable values is specified,
the default value and the value passed in must be one of these values, or None if `allow_none` is True.
If only the list of acceptable types is specified,
the default value and the value passed in must be an instance one of these types, or None if `allow_none` is True.
It is an error to attempt to specify both a list of acceptable values and a list of acceptable types.

.. tags:: Options
