"""Test the Newton nonlinear solver. """

import unittest
import warnings


import numpy as np

import openmdao.api as om
from openmdao.core.tests.test_discrete import InternalDiscreteGroup
from openmdao.test_suite.components.double_sellar import DoubleSellar, DoubleSellarImplicit, \
     SubSellar
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
     SellarNoDerivatives, SellarDerivatives, SellarStateConnection, StateConnection, \
     SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestNewton(unittest.TestCase):

    def test_specify_newton_linear_solver_in_system(self):

        my_newton = om.NewtonSolver(solve_subsystems=False)
        my_newton.linear_solver = om.DirectSolver()

        prob = om.Problem(model=SellarDerivatives(nonlinear_solver=my_newton))

        prob.setup()

        self.assertIsInstance(prob.model.nonlinear_solver.linear_solver, om.DirectSolver)

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_feature_newton_basic(self):
        """ Feature test for slotting a Newton solver and using it to solve
        Sellar.
        """
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(model=SellarDerivatives(nonlinear_solver=om.NewtonSolver(solve_subsystems=False)))

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_sellar_grouped(self):
        # Tests basic Newton solution on Sellar in a subgroup

        prob = om.Problem(model=SellarDerivativesGrouped(nonlinear_solver=om.NewtonSolver(solve_subsystems=False)))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

    def test_sellar(self):
        # Just tests Newton on Sellar with FD derivs.

        prob = om.Problem(model=SellarNoDerivatives(nonlinear_solver=om.NewtonSolver(solve_subsystems=False)))

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

    def test_sellar_derivs(self):
        # Test top level Sellar (i.e., not grouped).
        # Also, piggybacked testing that makes sure we only call apply_nonlinear
        # on the head component behind the cycle break.

        prob = om.Problem()
        prob.model = SellarDerivatives(nonlinear_solver=om.NewtonSolver(solve_subsystems=False),
                                       linear_solver=om.LinearBlockGS())

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

        ## Make sure we only call apply_linear on 'heads'
        #nd1 = prob.model.d1.execution_count
        #nd2 = prob.model.d2.execution_count
        #if prob.model.d1._run_apply == True:
            #self.assertEqual(nd1, 2*nd2)
        #else:
            #self.assertEqual(2*nd1, nd2)

    def test_sellar_derivs_with_Lin_GS(self):

        prob = om.Problem(model=SellarDerivatives(nonlinear_solver=om.NewtonSolver(solve_subsystems=False)))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

    def test_sellar_state_connection(self):
        # Sellar model closes loop with state connection instead of a cycle.

        prob = om.Problem(model=SellarStateConnection(nonlinear_solver=om.NewtonSolver(solve_subsystems=False)))

        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

    def test_sellar_state_connection_fd_system(self):
        # Sellar model closes loop with state connection instead of a cycle.
        # This test is just fd.
        prob = om.Problem(model=SellarStateConnection(nonlinear_solver=om.NewtonSolver(solve_subsystems=False)))

        prob.model.approx_totals(method='fd')

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 6)

    def test_sellar_specify_linear_solver(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        proms = ['x', 'z', 'y1', 'state_eq.y2_actual', 'state_eq.y2_command', 'd1.y2', 'd2.y2']
        sub = model.add_subsystem('sub', om.Group(), promotes=proms)

        subgrp = sub.add_subsystem('state_eq_group', om.Group(),
                                   promotes=['state_eq.y2_actual', 'state_eq.y2_command'])
        subgrp.linear_solver = om.ScipyKrylov()
        subgrp.add_subsystem('state_eq', StateConnection())

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

        model.connect('state_eq.y2_command', 'd1.y2')
        model.connect('d2.y2', 'state_eq.y2_actual')

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                               z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                           promotes=['x', 'z', 'y1', 'obj'])
        model.connect('d2.y2', 'obj_cmp.y2')

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
        model.connect('d2.y2', 'con_cmp2.y2')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        # Use bad settings for this one so that problem doesn't converge.
        # That way, we test that we are really using Newton's Lin Solver
        # instead.
        model.linear_solver = om.ScipyKrylov()
        model.linear_solver.options['maxiter'] = 1

        # The good solver
        model.nonlinear_solver.linear_solver = om.ScipyKrylov()

        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(model.nonlinear_solver._iter_count, 8)
        self.assertEqual(model.linear_solver._iter_count, 0)
        self.assertGreater(model.nonlinear_solver.linear_solver._iter_count, 0)

    def test_sellar_specify_linear_direct_solver(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        proms = ['x', 'z', 'y1', 'state_eq.y2_actual', 'state_eq.y2_command', 'd1.y2', 'd2.y2']
        sub = model.add_subsystem('sub', om.Group(), promotes=proms)

        subgrp = sub.add_subsystem('state_eq_group', om.Group(),
                                   promotes=['state_eq.y2_actual', 'state_eq.y2_command'])
        subgrp.linear_solver = om.ScipyKrylov()
        subgrp.add_subsystem('state_eq', StateConnection())

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

        model.connect('state_eq.y2_command', 'd1.y2')
        model.connect('d2.y2', 'state_eq.y2_actual')

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                               z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                           promotes=['x', 'z', 'y1', 'obj'])
        model.connect('d2.y2', 'obj_cmp.y2')

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
        model.connect('d2.y2', 'con_cmp2.y2')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        # Use bad settings for this one so that problem doesn't converge.
        # That way, we test that we are really using Newton's Lin Solver
        # instead.
        sub.linear_solver = om.ScipyKrylov()
        sub.linear_solver.options['maxiter'] = 1

        # The good solver
        model.nonlinear_solver.linear_solver = om.DirectSolver()

        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(model.nonlinear_solver._iter_count, 8)
        self.assertEqual(model.linear_solver._iter_count, 0)

    def test_solve_subsystems_basic(self):
        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = om.DirectSolver(assemble_jac=True)
        g1.options['assembled_jac_type'] = 'dense'

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver(assemble_jac=True)
        g2.options['assembled_jac_type'] = 'dense'

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        model.nonlinear_solver.options['solve_subsystems'] = True

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_basic_csc(self):
        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.options['assembled_jac_type'] = 'dense'
        g1.linear_solver = om.DirectSolver(assemble_jac=True)

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver(assemble_jac=True)
        g2.options['assembled_jac_type'] = 'dense'

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_basic_dense_jac(self):
        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_basic_dense_jac_scaling(self):
        prob = om.Problem(model=DoubleSellar(units=None, scaling=True))
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_basic_dense_jac_units_scaling(self):
        prob = om.Problem(model=DoubleSellar(units=True, scaling=True))
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.nonlinear_solver.linesearch = None
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.nonlinear_solver.linesearch = None
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.nonlinear_solver.linesearch = None
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.0533333333, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.0533333333, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_top(self):
        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_top_csc(self):
        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_top_implicit(self):
        prob = om.Problem(model=DoubleSellarImplicit())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_top_implicit_scaling(self):
        prob = om.Problem(model=DoubleSellarImplicit(scaling=True))
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_top_implicit_scaling_units(self):
        prob = om.Problem(model=DoubleSellarImplicit(units=True, scaling=True))
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.nonlinear_solver.linesearch = None
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.nonlinear_solver.linesearch = None
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.nonlinear_solver.linesearch = None
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.053333333, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.053333333, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_subgroup(self):
        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g1.linear_solver = om.DirectSolver(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, rtol=1.0e-5)
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.ScipyKrylov()

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_internals(self):
        # Here we test that this feature is doing what it should do by counting the
        # number of calls in various places.

        class CountNewton(om.NewtonSolver):
            """ This version of Newton also counts how many times it runs in total."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.options['solve_subsystems'] = True
                self.total_count = 0

            def _single_iteration(self):
                super()._single_iteration()
                self.total_count += 1

        class CountDS(om.DirectSolver):
            """ This version of Newton also counts how many times it linearizes"""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.lin_count = 0

            def _linearize(self):
                super()._linearize()
                self.lin_count += 1

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        # each SubSellar group converges itself
        g1 = model.g1
        g1.nonlinear_solver = CountNewton()
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = CountDS()  # used for derivatives

        g2 = model.g2
        g2.nonlinear_solver = CountNewton()
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()

        # Enfore behavior: max_sub_solves = 0 means we run once during init

        model.nonlinear_solver.options['maxiter'] = 5
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        # Verifying subsolvers ran
        self.assertEqual(g1.nonlinear_solver.total_count, 2)
        self.assertEqual(g2.nonlinear_solver.total_count, 2)
        self.assertEqual(g1.linear_solver.lin_count, 2)

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        # each SubSellar group converges itself
        g1 = model.g1
        g1.nonlinear_solver = CountNewton()
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = CountDS()  # used for derivatives

        g2 = model.g2
        g2.nonlinear_solver = CountNewton()
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()

        # Enforce Behavior: baseline

        model.nonlinear_solver.options['maxiter'] = 5
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 5
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        # Verifying subsolvers ran
        self.assertEqual(g1.nonlinear_solver.total_count, 5)
        self.assertEqual(g2.nonlinear_solver.total_count, 5)
        self.assertEqual(g1.linear_solver.lin_count, 5)

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        # each SubSellar group converges itself
        g1 = model.g1
        g1.nonlinear_solver = CountNewton()
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = CountDS()  # used for derivatives

        g2 = model.g2
        g2.nonlinear_solver = CountNewton()
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()

        # Enfore behavior: max_sub_solves = 1 means we run during init and first iteration of iter_execute

        model.nonlinear_solver.options['maxiter'] = 5
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 1
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        # Verifying subsolvers ran
        self.assertEqual(g1.nonlinear_solver.total_count, 4)
        self.assertEqual(g2.nonlinear_solver.total_count, 4)
        self.assertEqual(g1.linear_solver.lin_count, 4)

    def test_maxiter_one(self):
        # Fix bug when maxiter was set to 1.
        # This bug caused linearize to run before apply in this case.

        class ImpComp(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)
                self.applied = False

                self.declare_partials(of='*', wrt='*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = np.exp(outputs['x']) - \
                    inputs['a']**2 * outputs['x']**2
                self.applied = True

            def solve_nonlinear(self, inputs, outputs):
                pass

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = np.exp(outputs['x']) - \
                    2 * inputs['a']**2 * outputs['x']
                jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2

                if not self.applied:
                    raise RuntimeError("Bug! Linearize called before Apply!")

        prob = om.Problem()
        root = prob.model
        root.add_subsystem('p1', om.IndepVarComp('a', 1.0))
        root.add_subsystem('comp', ImpComp())
        root.connect('p1.a', 'comp.a')

        root.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        root.nonlinear_solver.options['maxiter'] = 1
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

    def test_err_on_non_converge(self):
        # Raise AnalysisError when it fails to converge

        prob = om.Problem()
        nlsolver = om.NewtonSolver(solve_subsystems=False)
        prob.model = SellarDerivatives(nonlinear_solver=nlsolver,
                                       linear_solver=om.LinearBlockGS())

        nlsolver.options['err_on_non_converge'] = True
        nlsolver.options['maxiter'] = 1

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(om.AnalysisError) as context:
            prob.run_driver()

        msg = "Solver 'NL: Newton' on system '' failed to converge in 1 iterations."
        self.assertEqual(str(context.exception), msg)

    def test_reraise_child_analysiserror(self):
        # Raise AnalysisError when it fails to converge

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver()
        g1.nonlinear_solver.options['maxiter'] = 1
        g1.nonlinear_solver.options['err_on_non_converge'] = True
        g1.nonlinear_solver.options['solve_subsystems'] = True
        g1.linear_solver = om.DirectSolver(assemble_jac=True)

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver()
        g2.nonlinear_solver.options['maxiter'] = 1
        g2.nonlinear_solver.options['err_on_non_converge'] = True
        g2.nonlinear_solver.options['solve_subsystems'] = True
        g2.linear_solver = om.DirectSolver(assemble_jac=True)

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['err_on_non_converge'] = True
        model.nonlinear_solver.options['reraise_child_analysiserror'] = True

        prob.setup()

        with self.assertRaises(om.AnalysisError) as context:
            prob.run_model()

        msg = "Solver 'NL: Newton' on system 'g1' failed to converge in 1 iterations."
        self.assertEqual(str(context.exception), msg)

    def test_err_message_inf_nan(self):

        prob = om.Problem()
        nlsolver = om.NewtonSolver(solve_subsystems=False)
        prob.model = SellarDerivatives(nonlinear_solver=nlsolver,
                                       linear_solver=om.LinearBlockGS())

        nlsolver.options['err_on_non_converge'] = True
        nlsolver.options['maxiter'] = 1

        prob.setup()
        prob.set_solver_print(level=0)

        prob['x'] = np.nan

        with self.assertRaises(om.AnalysisError) as context:
            prob.run_model()

        msg = "Solver 'NL: Newton' on system '': residuals contain 'inf' or 'NaN' after 0 iterations."
        self.assertEqual(str(context.exception), msg)

    def test_relevancy_for_newton(self):

        class TestImplCompSimple(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)

                self.declare_partials(of='*', wrt='*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = np.exp(outputs['x']) - \
                    inputs['a']**2 * outputs['x']**2

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = np.exp(outputs['x']) - \
                    2 * inputs['a']**2 * outputs['x']
                jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2


        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        model.add_subsystem('icomp', TestImplCompSimple())
        model.add_subsystem('ecomp', om.ExecComp('y = x*p', p=1.0))

        model.connect('p1.x', 'ecomp.x')
        model.connect('icomp.x', 'ecomp.p')

        model.add_design_var('p1.x', 3.0)
        model.add_objective('ecomp.y')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.ScipyKrylov()

        prob.setup()

        prob.run_model()

        J = prob.compute_totals()
        assert_near_equal(J['ecomp.y', 'p1.x'][0][0], -0.703467422498, 1e-6)

    def test_error_specify_solve_subsystems(self):
        # Raise AnalysisError when it fails to converge

        prob = om.Problem()
        model = prob.model

        model.nonlinear_solver = om.NewtonSolver()

        prob.setup()

        with self.assertRaises(ValueError) as context:
            prob.run_model()

        msg = "NewtonSolver in <model> <class Group>: solve_subsystems must be set by the user."
        self.assertEqual(str(context.exception), msg)



class TestNewtonFeatures(unittest.TestCase):

    def test_feature_basic(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_feature_maxiter(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = om.DirectSolver()

        newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['maxiter'] = 2

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.5878516779, .00001)
        assert_near_equal(prob.get_val('y2'), 12.0607416105, .00001)

    def test_feature_rtol(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = om.DirectSolver()

        newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['rtol'] = 1e-3

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.5878516779, .00001)
        assert_near_equal(prob.get_val('y2'), 12.0607416105, .00001)

    def test_feature_atol(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = om.DirectSolver()

        newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['atol'] = 1e-4

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.5882856302, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_feature_linear_solver(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
             SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = om.LinearBlockGS()

        newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        newton.linear_solver = om.DirectSolver()

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_feature_max_sub_solves(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('g1', SubSellar())
        model.add_subsystem('g2', SubSellar())

        model.connect('g1.y2', 'g2.x')
        model.connect('g2.y2', 'g1.x')

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.DirectSolver()

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()

        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        prob.setup()
        prob.run_model()

    def test_feature_err_on_non_converge(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = om.DirectSolver()

        newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['maxiter'] = 1
        newton.options['err_on_non_converge'] = True

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        try:
            prob.run_model()
        except om.AnalysisError:
            pass

    def test_solve_subsystems_basic(self):
        import openmdao.api as om
        from openmdao.test_suite.components.double_sellar import DoubleSellar

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()

        model.nonlinear_solver.options['solve_subsystems'] = True

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('g1.y1'), 0.64, .00001)
        assert_near_equal(prob.get_val('g1.y2'), 0.80, .00001)
        assert_near_equal(prob.get_val('g2.y1'), 0.64, .00001)
        assert_near_equal(prob.get_val('g2.y2'), 0.80, .00001)


if __name__ == "__main__":
    unittest.main()
