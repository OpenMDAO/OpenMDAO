.. _component_metadata:

********************************************
Component metadata (arguments to components)
********************************************

The primary job of a component, whether explicit or implicit, is to define inputs and outputs,
as well as the mapping that computes the outputs given the inputs.
Often, however, there are incidental parameters that affect the behavior of the component
but are not considered input variables in the sense of being computed as an output of another component.

OpenMDAO provides a way of declaring these parameters, which are referred to as *metadata*.
They are declared in the 'initialize' method of the user's component,
and the values are typically passed in upon instantiation of the component.
Once the values are passed in during instantiation, they are automatically set into the metadata object
and are available for use in any method other than 'initialize'.

Alternatively, the values can be set at any time, whether in any method of the component
(except for 'initialize') or outside of the component definition after the component is instantiated.
Metadata can be declared along with their default values and various checks for validity,
such as a list of acceptable values or types.
The full list of options can be found in the method signature at the bottom of this page.

A simple example
----------------

A common use of metadata is to specify the shape or size of the component's input and output variables.

.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.VectorDoublingComp

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_simple

In this example, 'size' is required; the first time we try to get 'size',
we would have got an error if we

1. did not pass in 'size' when instantiating 'VectorDoublingComp' and
2. did not set its value in the code for VectorDoublingComp.

Not setting a default value when declaring it implies that the value must be set prior to getting it.

Providing default values
------------------------

One reason why using metadata is convenient is that a default value can be specified,
making it optional to pass the value in during component instantiation.

.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.LinearCombinationComp

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_with_default

In this example, both 'a' and 'b' are optional, so it is valid to pass in 'a', but not 'b'.

Specifying values and types
---------------------------

Another commonly-used metadata feature is specifying acceptable values and types.
If only the list of acceptable values is specified,
the default value and the value passed in must be one of these values.
If only the list of acceptable types is specified,
the default value and the value passed in must be an instance one of these types.
If both the lists of acceptable values and types are specified,
the default value and the value passed in must be one of the values OR an instance one of the types.
This is illustrated in the following example.

.. embed-code::
    openmdao.utils.tests.test_options_dictionary_feature.UnitaryFunctionComp

.. embed-test::
    openmdao.utils.tests.test_options_dictionary_feature.TestOptionsDictionaryFeature.test_values_and_types

In this example, it is convenient to specify a list of acceptable values of the 'func' metadata,
but also to provide a valid type if it is not one of these acceptable values.


Method Signature
----------------

.. automethod:: openmdao.utils.options_dictionary.OptionsDictionary.declare
    :noindex:


.. tags:: Metadata
