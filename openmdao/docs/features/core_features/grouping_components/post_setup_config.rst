
***********************************
Changing Model Settings After Setup
***********************************

After Problem :code:`setup` has been called, the entire model hierarchy has been instantiated and
:ref:`setup and configure <feature_configure>` have been called on all Groups and Components.
However, you may still want to make some changes to your model configuration.

OpenMDAO allows you to do a limited number of things after the Problem :code:`setup` has been called, but before
you have called :code:`run_model` or :code:`run_driver`. These allowed actions include the following:

 - :ref:`Set initial conditions for unconnected inputs or states <set-and-get-variables>`
 - :ref:`Assign linear and nonlinear solvers <feature_solvers>`
 - Change solver settings
 - Assign Dense or Sparse Jacobians
 - :ref:`Set execution order <feature_set_order>`
 - Assign case recorders


Here, we instantiate a hierarchy of Groups, and then change the solver to one that can solve this problem.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_post_setup_solver_configure
    :layout: code, output

.. tags:: Group, System