.. _`api_translation`:

******************************************
Upgrading from OpenMDAO 2.10 to OpenMDAO 3
******************************************

In the OpenMDAO 3.0 release, a few changes were made to the API.  In addition, we removed all
deprecation warnings and fully deprecated the old behavior for all API changes that were made
over the lifespan of OpenMDAO 2.x.  The changes are all summarized here.


Building Component Models
-------------------------

Declare a Component with distributed variables
==============================================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.components.distributed_components.DistribComp
      DistribComp
      distributed

    class DistribComp(ExplicitComponent):

        def __init__(self, size):
            super(DistribComp, self).__init__()
            self.distributed = True


Declare a variable that is explicitly unitless
==============================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_group.TestConnect.test_connect_units_with_unitless
      None
      None

    prob.model.add_subsystem('tgt', om.ExecComp('y = 3 * x', x={'units': 'unitless'}))


Add a subsystem to a Group
==========================

.. content-container ::

  .. embed-compare::
      openmdao.test_suite.test_examples.basic_opt_paraboloid.BasicOptParaboloid.test_constrained
      add_subsystem
      add_subsystem

    indeps = prob.model.add('indeps', om.IndepVarComp())


Add a linear or nonlinear solver to a Group
===========================================

.. content-container ::

  .. embed-compare::
     openmdao.test_suite.test_examples.test_circuit_analysis_derivs.TestNonlinearCircuit.test_nonlinear_circuit_analysis
      nonlinear_solver
      DirectSolver

    self.nl_solver = om.NewtonSolver()
    self.ln_solver = om.DirectSolver()


Declare an option with an explicit type
=======================================

.. content-container ::

  .. embed-compare::
      openmdao.components.vector_magnitude_comp.VectorMagnitudeComp
      initialize
      computed

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', type_=int, default=1,
                             desc='The number of points at which the vector magnitude is computed')


Component Library
-----------------

Create an interpolating component using Akima spline with uniform grid
======================================================================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_2to3doc_fixed_grid
      ycp
      run_model

    ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
    ncp = len(ycp)
    n = 11

    prob = om.Problem()

    comp = om.AkimaSplineComp(num_control_points=ncp, num_points=n,
                              name='chord')

    prob.model.add_subsystem('comp1', comp)

    prob.setup()
    prob['akima.chord:y_cp'] = ycp.reshape((1, ncp))
    prob.run_model()


Create an interpolating component using Akima spline with custom grid
=====================================================================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_basic_example
      xcp
      run_model

    xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
    ncp = len(xcp)
    n = 50
    x = np.linspace(1.0, 12.0, n)

    prob = om.Problem()

    comp = om.AkimaSplineComp(num_control_points=ncp, num_points=n,
                              name='chord', input_x=True,
                              input_xcp=True)

    prob.model.add_subsystem('akima', comp)

    prob.setup(force_alloc_complex=True)

    prob['akima.chord:x_cp'] = xcp
    prob['akima.chord:y_cp'] = ycp.reshape((1, ncp))
    prob['akima.chord:x'] = x

    prob.run_model()


Create an interpolating component using Bsplines
================================================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_bsplines_2to3doc
      sine_distribution
      run_model

    prob = om.Problem()
    model = prob.model

    n_cp = 5
    n_point = 10

    t = np.linspace(0, 0.5*np.pi, n_cp)
    x = np.empty((2, n_cp))
    x[0, :] = np.sin(t)
    x[1, :] = 2.0*np.sin(t)

    comp = om.BsplinesComp(num_control_points=n_cp,
                           num_points=n_point,
                           bspline_order=4,
                           distribution='sine',
                           vec_size=2,
                           in_name='h_cp',
                           out_name='h')

    model.add_subsystem('interp', comp)

    prob.setup()
    prob.run_model()


Create an ExecComp with diagonal partials
=========================================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_has_diag_partials
      ExecComp
      np.ones

    model.add_subsystem('comp', ExecComp('y=3.0*x + 2.5',
                                         vectorize=True,
                                         x=np.ones(5), y=np.ones(5)))


Create an IndepVarComp with multiple outputs
============================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_indep_var_comp.TestIndepVarComp.test_add_output
      IndepVarComp
      indep_var_2

    comp = om.IndepVarComp((
        ('indep_var_1', 1.0, {'lower': 0, 'upper': 10}),
        ('indep_var_2', 2.0, {'lower': 1., 'upper': 20}),
    ))


Create an ExternalCode
======================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_external_code_comp.ParaboloidExternalCodeCompDerivs
      ParaboloidExternalCodeCompDerivs
      ParaboloidExternalCodeCompDerivs

    class ParaboloidExternalCodeCompDerivs(om.ExternalCode):


Create a KSComponent
====================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_ks_comp.TestKSFunctionFeatures.test_basic
      KSComp
      KSComp

    model.add_subsystem('ks', om.KSComponent(width=2))


Create a MetaModel
==================

.. content-container ::

  .. embed-compare::
     openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_kriging
      MetaModelUnStructuredComp
      MetaModelUnStructuredComp

    sin_mm = om.MetaModel()


Create a MetaModelUnstructured
==============================

.. content-container ::

  .. embed-compare::
     openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_kriging
      MetaModelUnStructuredComp
      MetaModelUnStructuredComp

    sin_mm = om.MetaModelUnstructured()


Create a MetaModelStructured
============================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_meta_model_structured_comp.TestMetaModelStructuredCompFeature.test_vectorized
      MetaModelStructuredComp
      MetaModelStructuredComp

    interp = om.MetaModelStructured(method='scipy_cubic', vec_size=2)


Create a MultiFiMetaModel
=========================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_multifi_meta_model_unstructured_comp.MultiFiMetaModelFeatureTestCase.test_2_input_2_fidelity
      MultiFiMetaModelUnStructuredComp
      MultiFiMetaModelUnStructuredComp

    mm = om.MultiFiMetaModel(nfi=2)


Create a MultiFiMetaModelUnStructured
=====================================

.. content-container ::

  .. embed-compare::
      openmdao.components.tests.test_multifi_meta_model_unstructured_comp.MultiFiMetaModelFeatureTestCase.test_2_input_2_fidelity
      MultiFiMetaModelUnStructuredComp
      MultiFiMetaModelUnStructuredComp

    mm = om.MultiFiMetaModelUnStructured(nfi=2)


Add a FloatKrigingSurrogate to a MetaModelStructuredComp
========================================================

.. content-container ::

  .. embed-compare::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_kriging
      KrigingSurrogate
      KrigingSurrogate

    sin_mm.add_output('f_x', 0., surrogate=om.FloatKrigingSurrogate())


Specify a default surrogate model for MetaModelStructuredComp
=============================================================

.. content-container ::

  .. embed-compare::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelTestCase.test_metamodel_feature_vector2d
      KrigingSurrogate
      KrigingSurrogate

    trig = om.MetaModelUnStructuredComp(vec_size=size)
    trig.default_surrogate = om.KrigingSurrogate()


Solvers
-------

Declare a NewtonSolver with solve_subsystems set to False
=========================================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_linear_solver
      solve_subsystems
      solve_subsystems

    newton = model.nonlinear_solver = om.NewtonSolver()


Control how a solver handles an error raised in a subsolver
===========================================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_err_on_non_converge
      NewtonSolver
      err_on_non_converge

    newton = model.nonlinear_solver = NewtonSolver()
    newton.options['maxiter'] = 1
    newton.options['err_on_maxiter'] = True


Declare a BroydenSolver with the BoundsEnforceLS line search
============================================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.nonlinear.tests.test_broyden.TestBryodenFeature.test_circuit_options
      om.Broyden
      Broyden

    model.circuit.nonlinear_solver = om.BroydenSolver()
    model.circuit.nonlinear_solver.linesearch = om.BoundsEnforceLS()


Declare a NewtonSolver with the BoundsEnforceLS line search
===========================================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.nonlinear.tests.test_newton.TestNewtonFeatures.test_feature_rtol
      NewtonSolver
      NewtonSolver

    newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    newton.linesearch = om.BoundsEnforceLS()


Add a preconditioner to PETScKrylov
===================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_specify_precon
      PETScKrylov
      LinearBlockGS

    model.linear_solver = om.PETScKrylov()

    model.linear_solver.preconditioner = om.LinearBlockGS()


Add a preconditioner to ScipyKrylov
===================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linear.tests.test_scipy_iter_solver.TestScipyKrylovFeature.test_specify_precon
      linear_solver.precon
      linear_solver.precon

    model.linear_solver.preconditioner = om.LinearBlockGS()


Add a ArmijoGoldsteinLS to a NewtonSolver
=========================================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linesearch.tests.test_backtracking.TestFeatureLineSearch.test_feature_goldstein
      Newton
      ArmijoGoldsteinLS

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        ls = top.model.nonlinear_solver.line_search = om.ArmijoGoldsteinLS(bound_enforcement='vector')


Create a NonLinearRunOnce
=========================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.nonlinear.tests.test_nonlinear_runonce.TestNonlinearRunOnceSolver.test_feature_solver
      NonlinearRunOnce
      NonlinearRunOnce

    model.nonlinear_solver = om.NonLinearRunOnce()


Create a PetscKSP
=================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linear.tests.test_petsc_ksp.TestPETScKrylovSolverFeature.test_specify_solver
      PETScKrylov
      PETScKrylov

    model.linear_solver = om.PetscKSP()


Create a ScipyIterativeSolver
=============================

.. content-container ::

  .. embed-compare::
      openmdao.solvers.linear.tests.test_scipy_iter_solver.TestScipyKrylovFeature.test_specify_solver
      ScipyKrylov
      ScipyKrylov

    model.linear_solver = om.ScipyIterativeSolver()


Drivers
-------

Activate dynamic coloring on a Driver
=====================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_coloring.SimulColoringScipyTestCase.test_simul_coloring_example
      declare_coloring
      declare_coloring

    p.driver.options['dynamic_simul_derivs'] = True


Add a ScipyOptimizer to a Problem
=================================

.. content-container ::

  .. embed-compare::
      openmdao.drivers.tests.test_scipy_optimizer.TestScipyOptimizeDriverFeatures.test_feature_basic
      ScipyOptimizeDriver
      ScipyOptimizeDriver

    prob.driver = om.ScipyOptimizer()


Working with Derivatives
------------------------

Use a pre-computed coloring on a model
======================================

.. content-container ::

  .. embed-compare::
      openmdao.core.tests.test_coloring.SimulColoringConfigCheckTestCase._build_model
      use_fixed_coloring
      use_fixed_coloring

    p.driver.set_simul_deriv_color()


Case Reading
------------

Query the iteration coordinate for a case
=========================================

.. content-container ::

  .. embed-compare::
      openmdao.recorders.tests.test_sqlite_reader.TestSqliteCaseReader.test_linesearch
      CaseReader
      case.name

    cr = om.CaseReader(self.filename)

    for i, c in enumerate(cr.list_cases()):
        case = cr.get_case(c)

        coord = case.iteration_coordinate


Running a Model
---------------

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
