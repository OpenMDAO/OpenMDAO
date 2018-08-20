.. _`api_translation`:

*************************
Upgrading from OpenMDAO 1
*************************

This guide takes how you did things in OpenMDAO 1 and shows how to do them OpenMDAO 2.
It is not a comprehensive guide to using OpenMDAO 2, but focuses only on the things that have changed in the API.


Build a Model
-------------

Define an Explcit Component
===========================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.components.paraboloid.Paraboloid

    class Paraboloid(Component):
        """
        Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
        """

        def __init__(self):
            super(Paraboloid, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', val=0.0)

        def solve_nonlinear(self, params, unknowns, resids):
            """
            f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

            Optimal solution (minimum): x = 6.6667; y = -7.3333
            """
            x = params['x']
            y = params['y']

            unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        def linearize(self, params, unknowns, resids):
            """
            Jacobian for our paraboloid.
            """
            x = params['x']
            y = params['y']

            J = {}
            J['f_xy', 'x'] = 2.0*x - 6.0 + y
            J['f_xy', 'y'] = 2.0*y + 8.0 + x

            return J


Define an Implicit Component
============================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.components.implicit_newton_linesearch.ImplCompOneState
      ImplCompOneState
      10.0*exp

    class ImplCompOneState(Component):
        """
        A Simple Implicit Component

        R(x,y) = 0.5y^2 + 2y + exp(-16y^2) + 2exp(-5y) - x

        Solution:
        x = 1.2278849186466743
        y = 0.3968459
        """

        def setup(self):
            self.add_param('x', 1.2278849186466743)
            self.add_state('y', val=1.0)

        def apply_nonlinear(self, params, unknowns, resids):
            """
            Don't solve; just calculate the residual.
            """
            x = params['x']
            y = unknowns['y']

            resids['y'] = 0.5*y*y + 2.0*y + exp(-16.0*y*y) + 2.0*exp(-5.0*y) - x

        def linearize(self, params, unknowns, resids):
            """
            Analytical derivatives.
            """
            y = unknowns['y']

            J = {}

            # State equation
            J[('y', 'x')] = -1.0
            J[('y', 'y')] = y + 2.0 - 32.0*y*exp(-16.0*y*y) - 10.0*exp(-5.0*y)

            return J

Input-Input connections
============================

See more details in the doc for :ref:`add_subsystem() <feature_adding_subsystem_to_a_group>`.

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_input_input
      Problem
      run_model

    prob = Problem()
    root = prob.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))

    root.add('comp1', Paraboloid())
    root.add('comp2', Paraboloid())

    #input-input connection
    root.connect('comp1.x', 'comp2.x')
    #then connect the indep var to just one of the inputs
    root.connect('p1.x', 'comp1.x')

    prob.setup()
    prob.run()


Run a Model
-----------

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


Print All Solver Messages
==========================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linesearch.tests.test_backtracking.TestFeatureLineSearch.test_feature_print_bound_enforce
      set_solver_print
      set_solver_print

    top.print_all_convergence(level=2)


Check a Model
-------------

Specify Finite Difference for all Component Derivatives
=======================================================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.components.sellar_feature.SellarDis1.setup

    def __init__(self):
        super(SellarDis1, self).__init__()

        # Global Design Variable
        self.add_param('z', val=np.zeros(2))

        # Local Design Variable
        self.add_param('x', val=0.)

        # Coupling parameter
        self.add_param('y2', val=1.0)

        # Coupling output
        self.add_output('y1', val=1.0)

        # Finite difference all partials.
        self.deriv_options['type'] = 'fd'


Specify FD Form and Stepsize on Specific Derivatives
====================================================

.. content-container ::

  .. embed-compare::
      openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_fd_options
      setup
      central

    def __init__(self):
        super(PartialComp, self).__init__()

        self.add_param('x', shape=(4,), step_size=1e-4, form='backward')
        self.add_param('y', shape=(2,), step_size=1e-6, form='central')
        self.add_param('y2', shape=(2,), step_size=1e-6, form='central')
        self.add_output('f', shape=(2,))


Check Partial Derivatives on All Components
===========================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_incorrect_jacobian
      check_partials
      check_partials

      data = prob.check_partials()


Check Partial Derivatives with Complex Step
===========================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_method_global
      opts
      check_partials

        prob.root.deriv_options['check_type'] = 'cs'

        prob.setup()
        prob.run()

        prob.check_partials()


Change Group Level Derivative Behavior
---------------------------------------

Force Group or Model to use Finite Difference
=============================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_basic
      approx_totals
      approx_totals

      model.deriv_options['type'] = 'fd'


Force Group or Model to use Finite Difference with Specific Options
===================================================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_approx_derivs.ApproxTotalsFeature.test_arguments
      approx_totals
      approx_totals

      model.deriv_options['type'] = 'fd'
      model.deriv_options['step_size'] = '1e-7'
      model.deriv_options['form'] = 'central'
      model.deriv_options['step_calc'] = 'relative'


Add Design Variables
--------------------

Add a Design Variable to a Model
================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_driver.TestDriver.test_basic_get
      Problem
      add_design_var

    prob = Problem()
    prob.root = SellarDerivatives()

    prob.add_desvar('z')


Add a Design Variable with Scale and Offset that Maps [3, 5] to [0, 1]
======================================================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_driver.TestDriver.test_scaled_design_vars
      Problem
      add_design_var

    prob = Problem()
    prob.root = SellarDerivatives()

    prob.add_desvar('z', scaler=0.5, adder=-3.0)


Set Solvers
-----------

Setup a Problem Using the PETScVector
=====================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_problem.TestProblem.test_feature_basic_setup
      vector_class
      vector_class

    prob.setup(impl=PetscImpl)


Specify Newton as a Nonlinear Solver in a Group
===============================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_basic
      NewtonSolver()
      NewtonSolver()

    model.nl_solver = Newton()


Specify Block Gauss-Seidel as a Nonlinear Solver in a Group
===========================================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_basic
      NonlinearBlockGS()
      NonlinearBlockGS()

    model.nl_solver = NLGaussSeidel()


Specify Scipy GMRES as a Linear Solver in a Group
=================================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linear.tests.test_scipy_iter_solver.TestScipyKrylovFeature.test_specify_solver
      ScipyKrylov()
      ScipyKrylov()

    model.ln_solver = ScipyGMRES()


Specify Linear Block Gauss-Seidel as a Linear Solver in a Group
===============================================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linear.tests.test_linear_block_gs.TestBGSSolverFeature.test_specify_solver
      LinearBlockGS()
      LinearBlockGS()

    model.ln_solver = LinearGaussSeidel()


Total Derivatives
-----------------


Computing Total Derivatives
===========================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_compute_totals
      prob.compute_totals
      prob.compute_totals

    prob.calc_gradient(indep_list=['p1.x', 'p2.y'], unknown_list=['comp.f_xy'])

Setting Derivative Computation Mode
===================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_problem.TestProblem.test_feature_simple_run_once_set_deriv_mode
      prob.setup
      prob.compute_totals

    root.ln_solver.options['mode'] = 'rev'
    # root.ln_solver.options['mode'] = 'fwd'
    # root.ln_solver.options['mode'] = 'auto'
    prob.setup()
    prob.run()
    prob.calc_gradient(indep_list=['p1.x', 'p2.y'], unknown_list=['comp.f_xy'])
