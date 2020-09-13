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
      style2

    prob = om.Problem()
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
    indeps.add_output('x', 3.0)
    indeps.add_output('y', -4.0)

    prob.model.add_subsystem('paraboloid',
                             om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

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
      style2

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


Declaring a New Name for a Promoted Input
=========================================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.tests.test_auto_ivc_api_conversion.TestConversionGuideDoc.test_promote_new_name
      Problem
      prob.setup
      style2

        prob = om.Problem()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('width', 3.0)
        indeps.add_output('length', -4.0)

        prob.model.add_subsystem('paraboloid',
                                 om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

        prob.model.connect('indeps.width', 'paraboloid.x')
        prob.model.connect('indeps.length', 'paraboloid.y')

        prob.setup()

Declare an Input Defined with Source Indices as a Design Variable
=================================================================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.tests.test_auto_ivc_api_conversion.TestConversionGuideDoc.test_promote_src_indices
      MyComp1
      run_model
      style2

        class MyComp1(om.ExplicitComponent):
            def setup(self):
                # this input will connect to entries 0, 1, and 2 of its source
                self.add_input('x', np.ones(3), src_indices=[0, 1, 2])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        class MyComp2(om.ExplicitComponent):
            def setup(self):
                # this input will connect to entries 3 and 4 of its source
                self.add_input('x', np.ones(2), src_indices=[3, 4])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*4.0

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp1(), promotes_inputs=['x'])
        p.model.add_subsystem('C2', MyComp2(), promotes_inputs=['x'])

        p.model.add_design_var('x')
        p.setup()
        p.run_model()


Setting Default Units for an Input
==================================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.tests.test_auto_ivc_api_conversion.TestConversionGuideDoc.test_units
      Problem
      setup
      style2

        prob = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output('x2', 100.0, units='degC')
        prob.model.add_subsystem('T1', ivc,
                                 promotes_outputs=['x2'])

        # Input units in degF
        prob.model.add_subsystem('tgtF', TgtCompF(),
                                 promotes_inputs=['x2'])

        # Input units in degC
        prob.model.add_subsystem('tgtC', TgtCompC(),
                                 promotes_inputs=['x2'])

        # Input units in deg
        prob.model.add_subsystem('tgtK', TgtCompK(),
                                 promotes_inputs=['x2'])

        prob.setup()



Creating a Distributed Component with Unconnected Inputs
========================================================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.tests.test_auto_ivc_api_conversion.TestConversionGuideDocMPI.test_prob_getval_dist_par
      size
      run_model

        size = 4

        prob = om.Problem()

        prob.model.add_subsystem("C1", DistribNoncontiguousComp(arr_size=size),
                                 promotes=['invec', 'outvec'])

        prob.setup()

        rank = prob.model.comm.rank
        if rank == 0:
            prob.set_val('invec', np.array([1.0, 3.0]))
        else:
            prob.set_val('invec', np.array([5.0, 7.0]))

        prob.run_model()


Setting and Getting Inputs
--------------------------

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.tests.test_auto_ivc_api_conversion.TestConversionGuideDoc.test_get_set
      Problem
      set_val
      style2

    prob = om.Problem()
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
    indeps.add_output('x', 3.0)
    indeps.add_output('y', -4.0)

    prob.model.add_subsystem('paraboloid',
                             om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

    prob.model.connect('indeps.x', 'paraboloid.x')
    prob.model.connect('indeps.y', 'paraboloid.y')

    prob.setup()

    x = prob.get_val('indeps.x')
    prob.set_val('indeps.y', 15.0)