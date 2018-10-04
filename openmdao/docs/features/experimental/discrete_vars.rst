
.. _discrete-variables:

******************
Discrete Variables
******************

There may be times when it's necessary to pass variables that are not floats or float arrays
between components.  These variables can be declared as discrete variables.  A discrete variable
can be any picklable python object.  It must be picklable in order to pass it between processes under
MPI.  If you never intend to use MPI with your model, you could get away with using
unpicklable discrete variables, but this isn't recommended, since it's difficult to know with
certainty that a given component will never be reused in an MPI context in the future.

In explicit and implicit components, the user must call :code:`add_discrete_input` and
:code:`add_discrete_output` to declare discrete variables in the :code:`setup` method.
An example is given below that shows a component that has a discrete input along with
continuous inputs and outputs.


.. embed-code::
    openmdao.core.tests.test_discrete.DiscreteFeatureTestCase.test_feature_discrete
    :layout: interleave

Method Signatures
-----------------

.. automethod:: openmdao.core.component.Component.add_discrete_input
    :noindex:

.. automethod:: openmdao.core.component.Component.add_discrete_output
    :noindex:


Discrete variables, like continuous ones, can be connected to each other using the :code:`connect`
function or by promoting an input and an output to the same name.  The type of the output
must be a valid subclass of the type of the input or the connection will raise an exception.

.. warning::
    If a model computes derivatives and any of those derivatives depend on the value of a discrete
    output variable, an exception will be raised.




