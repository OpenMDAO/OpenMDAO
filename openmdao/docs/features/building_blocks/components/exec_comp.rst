.. _feature_exec_comp:
.. index:: ExecComp Example

********
ExecComp
********


`ExecComp` is a component that provides a shortcut for building an ExplicitComponent that
represents a set of simple mathematical relationships between inputs and outputs. The ExecComp
automatically takes care of all of the component API methods, so you just need to instantiate
it with an equation.

ExecComp Constructor
--------------------

The call signature for the `ExecComp` constructor is:

.. automethod:: openmdao.components.exec_comp.ExecComp.__init__
    :noindex:

ExecComp Variable Metadata
--------------------------

The values of the `kwargs` can be `dicts` which define the initial value for the variables along with
other metadata. For example,

.. code-block:: python

    ExecComp('y=x', x={'units': 'ft'}, y={'units': 'm'})

Here is a list of the possible metadata that can be assigned to a variable in this way. The **Applies To** column indicates
whether the metadata is appropriate for input variables, output variables, or both.


================  ====================================================== ============================================================= ==============  ========
Name              Description                                            Valid Types                                                   Applies To      Default
================  ====================================================== ============================================================= ==============  ========
value             Initial value in user-defined units                    float, list, tuple, ndarray                                   input & output  1
shape             Variable shape, only needed if not an array            int, tuple, list, None                                        input & output  None
units             Units of variable                                      str, None                                                     input & output  None
desc              Description of variable                                str                                                           input & output  ""
res_units         Units of residuals                                     str, None                                                     output          None
ref               Value of variable when scaled value is 1               float, ndarray                                                output          1
ref0              Value of variable when scaled value is 0               float, ndarray                                                output          0
res_ref           Value of residual when scaled value is 1               float, ndarray                                                output          1
lower             Lower bound of variable                                float, list, tuple, ndarray, Iterable, None                   output          None
upper             Lower bound of variable                                float, list, tuple, ndarray, Iterable, None                   output          None
src_indices       Global indices of the variable                         int, list of ints, tuple of ints, int ndarray, Iterable, None input           None
flat_src_indices  If True, src_indices are indices into flattened source bool                                                          input           None
================  ====================================================== ============================================================= ==============  ========

These metadata are passed to the :code:`Component` methods :code:`add_input` and :code:`add_output`.
For more information about these metadata, see the documentation for the arguments to these Component methods:

- :meth:`add_input <openmdao.core.component.Component.add_input>`

- :meth:`add_output <openmdao.core.component.Component.add_output>`

ExecComp Usage Examples
-----------------------

For example, here is a simple component that takes the input and adds one to it.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_simple
    :layout: interleave

You can also create an ExecComp with multiple outputs by placing the expressions in a list.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_multi_output
    :layout: interleave

You can also declare an ExecComp with arrays for inputs and outputs, but when you do, you must also
pass in a correctly-sized array as an argument to the ExecComp call. This can be the initial value
in the case of unconnected inputs, or just an empty array with the correct size.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_array
    :layout: interleave

If all of your ExecComp's array inputs and array outputs are the same size and happen to have
diagonal partials, you can create a vectorized ExecComp by specifying a `vectorize=True` arg
to `__init__`.  This will cause the ExecComp to solve for its partials by complex stepping
all entries of an array input at once instead of looping over each entry individually.  Here's
a simple example:


.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_vectorize
    :layout: interleave


Functions from the math library are available for use in the expression strings.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_math
    :layout: interleave


You can also declare options like 'units', 'upper', or 'lower' on the inputs and outputs. Here is an example
where we declare all our inputs to be inches to trigger conversion from a variable expressed in feet in one
connection source.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_metadata
    :layout: interleave


.. tags:: ExecComp, Component, Examples
