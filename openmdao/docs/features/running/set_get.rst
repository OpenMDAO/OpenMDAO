Setting and Getting Component Variables
=========================================

You will both set and get the values in the dimensional and unscaled form via the <openmdao.core.problem.Problem> class.
If you have promoted both inputs and outputs to the same name,
then the output takes precedence and it determines the units you should work in.


Outputs and independent variables
-----------------------------------
To set or get the output variable, you reference it by its promoted name.
In the regular <openmdao.test_suite.components.sellar.SellarDerivatives> problem all the variables have been promoted to the top of the model.
So to get the value of the "y1" output defined in <openmdao.test_suite.components.sellar.SellarDis1withDerivatives> component you would do the following:

.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_promoted_sellar_set_get_outputs


You use the same syntax when working with the independent variables of your problem.
Independent variables hold values set by a user or are used as design variables by a <openmdao.core.driver.Driver>.
OpenMDAO requires that every variable must have an ultimate source, even independent variables.
We accomplish this by defining independent variables as outputs of a special component,
<openmdao.core.indepvarcomp.IndepVarComp>, that does not any inputs.
For example, consider our paraboloid tutorial problem problem which has two independent variables: `x` and `y`.

These would be defined and set as follows:

.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_set_indeps



As we said above, outputs are always referenced via their promoted name.
So if you built the Sellar problem using connections (see <openmdao.test_suite.components.sellar.SellarDerivativesConnected>),
instead of promoting everything, then you would access the variables like this:


.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_not_promoted_sellar_set_get_outputs


Working with Array Variables
------------------------------

When you have an array variables, for convenience we allow you to set the value with any properly sized array, list, or tuple.
In other words, the shape of the list has to match the shape of the actual data.


.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_set_get_array


Residuals
---------------------------
If you want to look at the residual values associated with any particular output variable, you will reference them using the same naming conventions the outputs.
Also like outputs, you will be given the residuals in the unscaled dimensional form.

.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_residuals


Inputs
------------------------------

.. note::
    99.9% of the time, you don't want to work with input variables.
    Instead you probably want to use the associated output variable.
    But if you really really want to, this is how you do it.

To set or get the and input variable, you reference it by its absolute path name. The full path name is necessary, because you could have a output (source) variable in units of meters and then two connected inputs (targets) in units of millimeters and centimeters respectively. Hence you need a specific path to reference each of two different inputs separately to get the value in that inputs units.


.. embed-python-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_promoted_sellar_set_get_inputs

Related Features
-----------------
building_components, setup, run_model