.. _`auto_ivc_api_translation`:

********************************************************
Conversion Guide for the Auto-IVC (IndepVarComp) Feature
********************************************************

As of the OpenMDAO 3.2 release. it is no longer necessary to add an IndepVarComp to your model
to handle the assignment of unconnected inputs as design variables.

Building Models
---------------

Declare Design Variables
========================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr
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