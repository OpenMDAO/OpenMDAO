.. _dynamic-shapes:

**************************************
Determining Variable Shapes at Runtime
**************************************


It's sometimes useful to create a component where the shapes of its inputs and/or outputs are
determined by their connections.  This allows us to create components representing general
purpose vector or matrix operations such as norms, summations, integrators, etc., that size
themselves appropriately based on the model that they're added to.

Turning on dynamic shape computation is straightforward.  You just specify `shape_by_conn`
and/or `copy_shape` in your `add_input` or `add_output` calls when you add variables
to your component.


shape_by_conn
-------------

Setting `shape_by_conn=True` when adding and input or output variable will allow the shape
of that variable to be determined at runtime based on the variable that connects to it.
For example:

.. code-block:: python

    self.add_input('x', shape_by_conn=True)


copy_shape
----------

Setting `copy_shape=<var_name>`, where `<var_name>` is the local name of another variable in your
component.  This will take the shape of the variable specified in `<var_name>` and use that
shape for the variable you're adding.  For example, the following will make the shape of the `y`
variable the same as the shape of the `x` variable.

.. code-block:: python

    self.add_output('y', copy_shape='x')


Note that `shape_by_conn` can be specified for outputs as well as for inputs, as can `copy_shape`.
This means that shape information can propagate through the model in either forward or reverse.
It's often advisable to specify both `shape_by_conn` and `copy_shape` for each
dynamically shaped variable, because that will allow your component's shapes to be resolved
whether constant shapes have been defined upstream or downstream of your component in the model.
For example, for our component with input `x` and output `y`, we could define our variables as
follows:

.. code-block:: python

    self.add_input('x', shape_by_conn=True, copy_shape='y')
    self.add_input('y', shape_by_conn=True, copy_shape='x')


This way, `y` can be used to determine the shape of `x`, or `x` can determine the shape of `y`.
