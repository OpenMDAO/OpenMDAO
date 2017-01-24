
..`Setting and Getting Variables`

Setting and Getting Component Variables
=========================================

You will both set and get the values in the dimensional and unscaled form via the <Problem> class.
If you have promoted both inputs and outputs to the same name,
then the output takes precedence and it determines the units you should work in.


Outputs and independent variables
-----------------------------------

To set or get the output variable, you reference it by its promoted name.
In the regular <SellarDerivatives> problem all the variables have been promoted to the top of the model.
So to get the value of the "y1" output defined in <SellarDis1withDerivatives> component you would do the following:

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_promoted_sellar_get_outputs


A somewhat special case to note is the output variables of <IndepVarComp>,
which define the independent variables of your problem.
OpenMDAO requires that every variable must have an ultimate source.
IndepVarComps provide this functionality for independent variables,
which are external inputs to the model that are set by the user or a <Driver>,
by having only output variables (i.e. no input variables). For example,
consider our paraboloid tutorial problem problem which has two independent variables: `x` and `y`.
These would be set as follows:

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_numpyvec_setup


As we said above, outputs are always referenced via their promoted name.
So if you built the Sellar problem using connections (see <SellarDerivativesConnected>),
instead of promoting everything, then you would access the variables like this:

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_not_promoted_sellar_get_outputs

Working with Array Variables
------------------------------


Inputs
------------------------------

.. note::
    99.9% of the time, you don't want to work with input variables.
    Instead you probably want to use the associated output variable.
    Getting inputs would be mostly necessary in debugging situations.

To set or get the and input variable, you reference it by its absolute path name. The full path name is necessary, because you could have a output (source) variable in units of meters and then two connected inputs (targets) in units of millimeters and centimeters respectively. Hence you need a specific path to reference each of two different inputs separately to get the value in that inputs units.

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_promoted_sellar_set_get_inputs

Related Features
-----------------
building_components, setup, run_model