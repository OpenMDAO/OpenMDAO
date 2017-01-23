Setting and Getting Variables
==============================

Internally OpenMDAO stores all data in an non-dimensional and scaled form, but via the :code:`Problem` class, you will both set and get the values in the dimensional and unscaled form.

.. note::
    before working with any variables, you must first call :code:`setup` on your problem.

If you want to work with value of an output (set or get it), you reference it by its promoted name. In the regular SellarDerivatives problem [TODO: link to SellarDerivatives definition page] all the variables have been promoted to the top of the model. So to get the value of the "y1" output defined in :code:`SellarDis1` component you would do the following:

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_simple_promoted_sellar_get


A somewhat special case to note is the outputs of :code:`IndepVarComp`, which are the independent variables of your problem. In OpenMDAO every variable must have an ultimate src, and IndepVarComps provide that for this special case. So while you might say that an independent variable is an input to your problem, the framework still references it by the location of its source, which is the promoted name of the output variable defined by the associated :code:`IndepVarComp`.


If you defined the Sellar problem using connections (instead of promoting everything),
then you would do this instead:




Examples
---------

These examples work with the Sellar problem [TODO: link to Sellar problem definition page]

.. embed-test:
    openmdao.core.tests.test_problem.TestProblem.test_feature_set_get


Related Features
-----------------
building_components, setup, run_model