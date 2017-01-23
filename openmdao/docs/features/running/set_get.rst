
..`Setting and Getting Variables`

Setting and Getting Variables
==============================

You will both set and get the values in the dimensional and unscaled form via the :code:`Problem` class.
If you have promoted both inputs and outputs to the same name,
then the output takes precedence and it determines the units you should work in.


Outputs and independent variables
-----------------------------------

To set or get the and output variable,
you reference it by its promoted name.
In the regular SellarDerivatives problem [TODO: link to SellarDerivatives definition page] all the variables have been promoted to the top of the model.
So to get the value of the "y1" output defined in :code:`SellarDis1` component you would do the following:

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_promoted_sellar_get_outputs


A somewhat special case to note is the outputs of :code:`IndepVarComp`,
which are the independent variables of your problem.
In OpenMDAO every variable must have an ultimate source and IndepVarComps provide that for this special case.
So while you might say that an independent variable is an input to your problem,
the framework still references it by the location of its source,
which is the promoted name of the output variable defined by the associated :code:`IndepVarComp`.


If you defined the Sellar problem using connections (instead of promoting everything),
then you would do this instead:

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_not_promoted_sellar_get_outputs


Inputs
------------------------------

.. note::
    99% of the time, you probably don't need to work with input variables.
    Instead you probably want to use the associated output variable.
    Getting inputs would be mostly necessary in debugging situations.

To set or get the and input variable, you reference it by its absolute path name. This is done to enable you to access the value of an input in **its** units. The full path name is necessary, because, for example, you could potentially have a output (source) variable in units of meters and then two connected inputs (targets) in units of millimeters and centimeters respectively. The absolute path name is unique and lets you specify which value you want.

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_promoted_sellar_set_get_inputs

Related Features
-----------------
building_components, setup, run_model