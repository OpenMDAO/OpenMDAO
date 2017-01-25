.. `Defining Independent Variables`

Defining independent variables
------------------------------

**Description:** Independent variables are those that are set externally to the model---therefore, they are called model inputs.
From the perspective of a component, they are component outputs that do not depend on any component inputs.
From the perspective of a model, they can be viewed as design variables or model parameters that are set by the user or driver, prior to running the model.

**Usage:** Independent variables are defined via the *IndepVarComp* class.
The *IndepVarComp* class is instantiated directly (without defining a subclass), passing in during instantiation the name, initial value, and other options of the variable(s).

**Examples:**

.. show-unittest-examples::
    indepvarcomp
