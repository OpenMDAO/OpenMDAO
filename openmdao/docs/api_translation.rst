.. _`api_translation`:

*********************
API Translation Guide
*********************

This guide takes how you did things in OpenMDAO Alpha and shows how to do them in the latest version of OpenMDAO.


Assemble and Run a Simple Model
===============================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_no_promote
      Problem
      run_model

    prob = Problem()
    root = prob.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('comp', Paraboloid())

    root.connect('p1.x', 'comp.x')
    root.connect('p2.y', 'comp.y')

    prob.setup()
    prob.run()


Run a Driver
============

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_driver.TestDriver.test_basic_get
      run_driver
      run_driver

    prob.run()


Run a Model without Running the Driver
======================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_no_promote
      run_model
      run_model

    prob.run_once()
