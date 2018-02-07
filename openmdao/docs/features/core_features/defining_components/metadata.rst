.. _component_metadata:

********************************************
Component Metadata (Arguments to Components)
********************************************

The primary jobs of a component, whether explicit or implicit, are to define inputs and outputs,
as well as to do the mapping that computes the outputs given the inputs.
Often, however, there are incidental parameters that affect the behavior of the component,
but which are not considered input variables in the sense of being computed as an output of another component.

OpenMDAO provides a way of declaring these parameters, which are contained in an
`OptionsDictionary` named *metadata* that is available in every component. Metadata
associated with a particular component must be declared in the `initialize` method
of the component definition. A default value can be provided as well as various checks
for validity, such as a list of acceptable values or types.

The full list of options is shown in the method signature below.

.. automethod:: openmdao.utils.options_dictionary.OptionsDictionary.declare
    :noindex:

Metadata values are typically passed at component instantiation time as keyword arguments,
which are automatically assigned into the metadata dictionary. The metadata is then available
for use in the component's other methods, such as `setup` and `compute`.

Alternatively, values can be set at a later time, in another method of the component
(except for `initialize`) or outside of the component definition after the component is
instantiated.

A Simple Example
----------------

Metadata is commonly used to specify the shape or size of the component's input and output 
variables, such as in this simple example.

.. embed-code::
    openmdao.test_suite.components.metadata_feature_vector

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple


Not setting a default value when declaring a metadata parameter implies that the value must be set by the user.

In this example, 'size' is required; We would have gotten an error if we:

1. Did not pass in 'size' when instantiating *VectorDoublingComp* and
2. Did not set its value in the code for *VectorDoublingComp*.


.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple_fail


Metadata Types
--------------

Metadata is not limited to simple types like :code:`int`.  In the following example, the
component takes a `Numpy` array as metadata:


.. embed-code::
    openmdao.test_suite.components.metadata_feature_array

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple_array


It is even possible to provide a function as metadata:


.. embed-code::
    openmdao.test_suite.components.metadata_feature_function

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple_function

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
the default value and the value passed in must be one of these values, or None if `allow_none` is True.
If only the list of acceptable types is specified,
the default value and the value passed in must be an instance one of these types, or None if `allow_none` is True.
It is an error to attempt to specify both a list of acceptable values and a list of acceptable types.

.. tags:: Metadata
