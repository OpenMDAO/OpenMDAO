.. _feature_exec_comp:
.. index:: ExecComp Example

********
ExecComp
********


`ExecComp` is a component that provides a shortcut for building an ExplicitComponent that
represents a set of simple mathematical relationships between inputs and outputs. The ExecComp
automatically takes care of all of the component API methods, so you just need to instantiate
it with an equation or a list of equations.

ExecComp Options
----------------

.. embed-options::
    openmdao.components.exec_comp
    ExecComp
    options

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

    ExecComp('xdot=x/t', x={'units': 'ft'}, t={'units': 's'}, xdot={'units': 'ft/s')

Here is a list of the possible metadata that can be assigned to a variable in this way. The **Applies To** column indicates
whether the metadata is appropriate for input variables, output variables, or both.


================  ====================================================== ============================================================= ==============  ========
Name              Description                                            Valid Types                                                   Applies To      Default
================  ====================================================== ============================================================= ==============  ========
value             Initial value in user-defined units                    float, list, tuple, ndarray                                   input & output  1
shape             Variable shape, only needed if not an array            int, tuple, list, None                                        input & output  None
shape_by_conn     Determine variable shape based on its connection       bool                                                          input & output  False
copy_shape        Determine variable shape based on named variable       str                                                           input & output  None
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
tags              Used to tag variables for later filtering              str, list of strs                                             input & output  None
================  ====================================================== ============================================================= ==============  ========

These metadata are passed to the :code:`Component` methods :code:`add_input` and :code:`add_output`.
For more information about these metadata, see the documentation for the arguments to these Component methods:

- :meth:`add_input <openmdao.core.component.Component.add_input>`

- :meth:`add_output <openmdao.core.component.Component.add_output>`

Registering User Functions
--------------------------

To get your own functions added to the internal namespace of ExecComp so you can call them
from within an ExecComp expression, you can use the :code:`ExecComp.register` function.

.. automethod:: openmdao.components.exec_comp.ExecComp.register
    :noindex:

Note that you're required, when registering a new function, to indicate whether that function
is complex safe or not.


ExecComp Example: Simple
------------------------

For example, here is a simple component that takes the input and adds one to it.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_simple
    :layout: interleave

ExecComp Example: Multiple Outputs
----------------------------------

You can also create an ExecComp with multiple outputs by placing the expressions in a list.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_multi_output
    :layout: interleave

ExecComp Example: Arrays
------------------------

You can declare an ExecComp with arrays for inputs and outputs, but when you do, you must also
pass in a correctly-sized array as an argument to the ExecComp call, or set the 'shape' metadata
for that variable as described earlier. If specifying the value directly, it can be the initial value
in the case of unconnected inputs, or just an empty array with the correct size.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_array
    :layout: interleave

ExecComp Example: Math Functions
--------------------------------

Functions from the math library are available for use in the expression strings.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_math
    :layout: interleave

ExecComp Example: Variable Properties
-------------------------------------

You can also declare properties like 'units', 'upper', or 'lower' on the inputs and outputs. In this
example we declare all our inputs to be inches to trigger conversion from a variable expressed in feet
in one connection source.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_metadata
    :layout: interleave

ExecComp Example: Diagonal Partials
-----------------------------------

If all of your ExecComp's array inputs and array outputs are the same size and happen to have
diagonal partials, you can make computation of derivatives for your ExecComp faster by specifying a
`has_diag_partials=True` argument
to `__init__` or via the component options. This will cause the ExecComp to solve for its partials
by complex stepping all entries of an array input at once instead of looping over each entry individually.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_has_diag_partials
    :layout: interleave

ExecComp Example: Options
-------------------------

Other options that can apply to all the variables in the component are variable shape and units.
These can also be set as a keyword argument in the constructor or via the component options. In the
following example the variables all share the same shape, which is specified in the constructor, and
common units that are specified by setting the option.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_options
    :layout: interleave


ExecComp Example: User function registration
--------------------------------------------

If the function is complex safe, then you don't need to do anything differently than you
would for any other ExecComp.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestFunctionRegistration.featuretest_register_simple
    :layout: interleave


ExecComp Example: Complex unsafe user function registration
-----------------------------------------------------------

If the function isn't complex safe, then derivatives involving that function
will have to be computed using finite difference instead of complex step.  The way to specify
that `fd` should be used for a given derivative is to call :code:`declare_partials`.

.. embed-code::
    openmdao.components.tests.test_exec_comp.TestFunctionRegistration.featuretest_register_simple_unsafe
    :layout: interleave


.. tags:: ExecComp, Component, Examples
