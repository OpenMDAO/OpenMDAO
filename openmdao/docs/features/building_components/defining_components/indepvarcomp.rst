:orphan:

.. `Defining independent variables`

Defining independent variables
------------------------------

**Description:** Independent variables are those that are set externally to the model---therefore, they are called model inputs.
From the perspective of a component, they are component outputs that do not depend on any component inputs.
From the perspective of a model, they can be viewed as design variables or model parameters that are set by the user or driver, prior to running the model.

**Usage:** Independent variables are defined via the *IndepVarComp* class.
The *IndepVarComp* class is instantiated directly (without defining a subclass), passing in during instantiation the name, initial value, and other options of the variable(s).

**Examples (this should be auto-generated):**

Define one independent variable and set its value.

::

    comp = IndepVarComp('indep_var')
    prob = Problem(comp).setup()

    self.assertEqual(prob['indep_var'], 1.0)

    prob['indep_var'] = 2.0
    self.assertEqual(prob['indep_var'], 2.0)

Define one independent variable with a default value.

::

    comp = IndepVarComp('indep_var', val=2.0)
    prob = Problem(comp).setup()

    self.assertEqual(prob['indep_var'], 2.0)

Define one independent array variable.

::

    array = numpy.array([
        [1., 2.],
        [3., 4.],
    ])

    comp = IndepVarComp('indep_var', val=array)
    prob = Problem(comp).setup()

    self.assertEqualArrays(prob['indep_var'], array)

.. showunittestexamples::
    indepvarcomp

Here's an example of embedding code, in the form of the TestIndepVarComp class:

.. embedPythonCode::
    openmdao.core.tests.test_component.TestIndepVarComp


Here's an example of embedding code, in the form of the `test___init___1var` method:

.. embedPythonCode::
    openmdao.core.tests.test_component.TestIndepVarComp.test___init___1var
    
