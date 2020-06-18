.. _`auto_ivc_api_translation`:

********************************************************
Conversion Guide for the Auto-IVC (IndepVarComp) Feature
********************************************************

As of the OpenMDAO 3.2 release. it is no longer necessary to add an IndepVarComp to your model
to handle the assignment of unconnected inputs as design variables.

Building Models
---------------

Declaring Design Variables
==========================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.tests.test_auto_ivc_api_conversion.TestConversionGuideDoc.test_tldr
      Problem
      run_driver

    prob = om.Problem()
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
    indeps.add_output('x', 3.0)
    indeps.add_output('y', -4.0)

    prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

    prob.model.connect('indeps.x', 'paraboloid.x')
    prob.model.connect('indeps.y', 'paraboloid.y')

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    prob.model.add_design_var('indeps.x', lower=-50, upper=50)
    prob.model.add_design_var('indeps.y', lower=-50, upper=50)
    prob.model.add_objective('paraboloid.f')

    prob.setup()
    prob.run_driver()


Declaring a Multi-Component Input as a Design Variable
======================================================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.tests.test_auto_ivc_api_conversion.TestConversionGuideDoc.test_constrained
      Problem
      run_driver

    prob = om.Problem()
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
    indeps.add_output('x', 3.0)
    indeps.add_output('y', -4.0)

    prob.model.add_subsystem('parab', Paraboloid())

    # define the component whose output will be constrained
    prob.model.add_subsystem('const', om.ExecComp('g = x + y'))

    prob.model.connect('indeps.x', ['parab.x', 'const.x'])
    prob.model.connect('indeps.y', ['parab.y', 'const.y'])

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'

    prob.model.add_design_var('indeps.x', lower=-50, upper=50)
    prob.model.add_design_var('indeps.y', lower=-50, upper=50)
    prob.model.add_objective('parab.f_xy')

    # to add the constraint to the model
    prob.model.add_constraint('const.g', lower=0, upper=10.)
    # prob.model.add_constraint('const.g', equals=0.)

    prob.setup()
    prob.run_driver()


Setting and Getting Inputs
--------------------------