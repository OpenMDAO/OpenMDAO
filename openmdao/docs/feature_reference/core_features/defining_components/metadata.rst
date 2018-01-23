.. _component_metadata:

********************************************
Component Metadata (Arguments to Components)
********************************************

The primary job of a component, whether explicit or implicit, is to define inputs and outputs,
as well as to do the mapping that computes the outputs given the inputs.
Often, however, there are incidental parameters that affect the behavior of the component,
but which are not considered input variables in the sense of being computed as an output of another component.

OpenMDAO provides a way of declaring these parameters, which are referred to as *metadata*.
They are declared in the 'initialize' method of the user's component,
and the values are typically passed in upon instantiation of the component.
Once the values are passed in during instantiation, they are automatically set into the metadata object
and are available for use in any method other than 'initialize'.

Alternatively, the values can be set at any time, whether in any method of the component
(except for 'initialize') or outside of the component definition after the component is instantiated.
Metadata can be declared along with their default values and various checks for validity,
such as a list of acceptable values or types.
The full list of options can be found in the method signature below.

.. automethod:: openmdao.utils.options_dictionary.OptionsDictionary.declare
    :noindex:

A Simple Example
----------------

Metadata is commonly used to specify the shape or size of the component's input and output variables.

.. embed-code::
    openmdao.test_suite.components.metadata_feature_vector

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple

In this example, 'size' is required; the first time we try to get 'size',
we would have gotten an error if we:

1. did not pass in 'size' when instantiating 'VectorDoublingComp' and
2. did not set its value in the code for VectorDoublingComp.

Not setting a default value when declaring it implies that the value must be set prior to getting it.

Providing Default Values
------------------------

One reason why using metadata is convenient is that a default value can be specified,
making it optional to pass the value in during component instantiation.

.. embed-code::
    openmdao.test_suite.components.metadata_feature_lincomb

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_with_default

In this example, both 'a' and 'b' are optional, so it is valid to pass in 'a', but not 'b'.

Specifying Values or Types
--------------------------

Another commonly-used metadata feature is specifying acceptable values or types.
If only the list of acceptable values is specified,
the default value and the value passed in must be one of these values, or None if allow_none is True.
If only the list of acceptable types is specified,
the default value and the value passed in must be an instance one of these types, or None if allow_none is True.
It is an error to attempt to specify both a list of acceptable values and a list of acceptable types.

.. tags:: Metadata
