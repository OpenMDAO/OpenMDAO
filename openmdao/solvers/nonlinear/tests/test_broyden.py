"""Test the Broyden nonlinear solver. """

import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.core.tests.test_distrib_derivs import DistribExecComp
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.components.sellar import SellarStateConnection, SellarDerivatives, \
     SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.utils.assert_utils import assert_near_equal, assert_warning

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None
from openmdao.utils.mpi import MPI


class VectorEquation(om.ImplicitComponent):
    """Equation with 5 states in a single vector. Should converge to x=[0,0,0,0,0]"""

    def setup(self):
        self.add_input('c', 0.01)
        self.add_output('x', np.ones((5, )))

    def apply_nonlinear(self, inputs, outputs, residuals):
        c = inputs['c']
        x = outputs['x']

        d = np.array([3, 2, 1.5, 1, 0.5])

        residuals['x'] = -d*x - c*x**3


class MixedEquation(om.ImplicitComponent):
    """Equation with 5 states split between 3 vars. Should converge to x=[0,0,0,0,0]"""

    def setup(self):
        self.add_input('c', 0.01)
        self.add_output('x12', np.ones((2, )))
        self.add_output('x3', 1.0)
        self.add_output('x45', np.ones((2, )))

        self.declare_partials(of=['x12', 'x3', 'x45'], wrt='c')
        self.declare_partials(of='x12', wrt='x12', rows=np.arange(2), cols=np.arange(2),
                              val=-np.array([3.0, 2]))
        self.declare_partials(of='x3', wrt='x3', rows=np.arange(1), cols=np.arange(1),
                              val=-np.array([1.5]))
        self.declare_partials(of='x45', wrt='x45', rows=np.arange(2), cols=np.arange(2),
                              val=-np.array([1, 0.5]))

    def apply_nonlinear(self, inputs, outputs, residuals):
        c = inputs['c']
        x = np.empty((5, ))
        x[:2] = outputs['x12']
        x[2] = outputs['x3']
        x[3:] = outputs['x45']

        d = np.array([3, 2, 1.5, 1, 0.5])

        res = -d*x - c*x**3
        residuals['x12'] = res[:2]
        residuals['x3'] = res[2]
        residuals['x45'] = res[3:]

    def linearize(self, inputs, outputs, jacobian):
        c = inputs['c']
        x = np.empty((5, ))
        x12 = outputs['x12']
        x3 = outputs['x3']
        x45 = outputs['x45']

        jacobian['x12', 'c'] = -3.0 * x12**2
        jacobian['x3', 'c'] = -3.0 * x3**2
        jacobian['x45', 'c'] = -3.0 * x45**2


class SpedicatoHuang(om.ImplicitComponent):

    cite = """
           @article{spedicato_hwang,
           author = {E. Spedicato, Z. Huang},
           title = {Numerical experience with newton-like methods for nonlinear algebraic systems},
           journal = {Computing},
           voluem = {86},
           year = {1997},
           }
           """

    def setup(self):

        self.n = 3

        self.add_input('x', np.array([0, 20]))
        self.add_output('y', 10.0*np.ones((self.n, )))

        self.declare_partials(of='y', wrt=['x', 'y'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']
        n = self.n

        residuals['y'][0] = y[0] + y[1] + x[0] + .25*(y[1] - x[0])**2
        residuals['y'][n-1] = y[n-1] + x[1] + y[n-2] + .25*(x[1] - y[n-2])**2
        for j in np.arange(1, n-1):
            residuals['y'][j] = y[j] + y[j+1] + y[j-1] + .25*(y[j+1] - y[j-1])**2

    def linearize(self, inputs, outputs, jacobian):
        x = inputs['x']
        y = outputs['y']
        n = self.n

        jacobian['y', 'x'][0, 0] = 1.0 - .5*(y[1] - x[0])
        jacobian['y', 'y'][0, 0] = 1.0
        jacobian['y', 'y'][0, 1] = 1.0 + .5*(y[1] - x[0])

        jacobian['y', 'x'][n-1, 1] = 1.0 + .5*(x[1] - y[n-2])
        jacobian['y', 'y'][n-1, n-1] = 1.0
        jacobian['y', 'y'][n-1, n-2] = 1.0 - .5*(x[1] - y[n-2])

        for j in np.arange(1, n-1):
            jacobian['y', 'y'][j, j-1] = 1.0 - .5*(y[j+1] - y[j-1])
            jacobian['y', 'y'][j, j] = 1.0
            jacobian['y', 'y'][j, j+1] = 1.0 + .5*(y[j+1] - y[j-1])


class TestBryoden(unittest.TestCase):

    def test_reraise_error(self):

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.BroydenSolver()
        g1.nonlinear_solver.options['maxiter'] = 1
        g1.nonlinear_solver.options['err_on_non_converge'] = True
        g1.linear_solver = om.DirectSolver(assemble_jac=True)

        g2 = model.g2
        g2.nonlinear_solver = om.BroydenSolver()
        g2.nonlinear_solver.options['maxiter'] = 1
        g2.nonlinear_solver.options['err_on_non_converge'] = True
        g2.linear_solver = om.DirectSolver(assemble_jac=True)

        model.nonlinear_solver = om.BroydenSolver()
        model.linear_solver = om.DirectSolver(assemble_jac=True)
        model.nonlinear_solver.options['err_on_non_converge'] = True
        model.nonlinear_solver.options['reraise_child_analysiserror'] = True

        prob.setup()

        with self.assertRaises(om.AnalysisError) as context:
            prob.run_model()

        msg = "Solver 'BROYDEN' on system 'g1' failed to converge in 1 iterations."
        self.assertEqual(str(context.exception), msg)

    def test_error_badname(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.options['state_vars'] = ['junk']

        with self.assertRaises(ValueError) as context:
            prob.run_model()

        msg = "BroydenSolver in <model> <class SellarStateConnection>: The following variable names were not found: junk"
        self.assertEqual(str(context.exception), msg)

    def test_error_need_direct_solver(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        with self.assertRaises(ValueError) as context:
            prob.run_model()

        msg = "BroydenSolver in <model> <class SellarStateConnection>: Linear solver must be DirectSolver when solving the full model."
        self.assertEqual(str(context.exception), msg)

    def test_simple_sellar(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

    def test_simple_sellar_cycle(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarDerivatives(nonlinear_solver=om.BroydenSolver(),
                                               linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.options['state_vars'] = ['y1']
        model.nonlinear_solver.options['compute_jacobian'] = True

        prob.set_solver_print(level=2)

        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['y2'], 12.05848819, .00001)

    def test_sellar_state_connection_fd_system(self):
        # Sellar model closes loop with state connection instead of a cycle.
        # This test is just fd.
        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())
        prob.model.approx_totals(method='fd')

        prob.setup()

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('state_eq.y2_command'), 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 6)

    def test_vector(self):
        # Testing Broyden on a 5 state single vector case.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('c', 0.01))
        model.add_subsystem('vec', VectorEquation())

        model.connect('p1.c', 'vec.c')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['vec.x']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup()

        prob.run_model()

        assert_near_equal(prob['vec.x'], np.zeros((5, )), 1e-6)

    def test_mixed(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12', 'mixed.x3', 'mixed.x45']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup()

        prob.run_model()

        assert_near_equal(prob['mixed.x12'], np.zeros((2, )), 1e-6)
        assert_near_equal(prob['mixed.x3'], 0.0, 1e-6)
        assert_near_equal(prob['mixed.x45'], np.zeros((2, )), 1e-6)

    def test_missing_state_warning(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup()

        msg = "The following states are not covered by a solver, and may have been " \
              "omitted from the BroydenSolver 'state_vars': mixed.x3, mixed.x45"

        with assert_warning(UserWarning, msg):
            prob.run_model()

        # Try again with promoted names.
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation(), promotes=['*'])

        model.connect('p1.c', 'c')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['x12']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup()

        msg = "The following states are not covered by a solver, and may have been " \
              "omitted from the BroydenSolver 'state_vars': x3, x45"

        with assert_warning(UserWarning, msg):
            prob.run_model()

    def test_mixed_promoted_vars(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation(), promotes_outputs=['x12', 'x3', 'x45'])

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['x12', 'x3', 'x45']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup()

        prob.run_model()

        assert_near_equal(prob['x12'], np.zeros((2, )), 1e-6)
        assert_near_equal(prob['x3'], 0.0, 1e-6)
        assert_near_equal(prob['x45'], np.zeros((2, )), 1e-6)

    def test_mixed_jacobian(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12', 'mixed.x3', 'mixed.x45']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.linear_solver = om.DirectSolver()

        prob.setup()

        prob.run_model()

        assert_near_equal(prob['mixed.x12'], np.zeros((2, )), 1e-6)
        assert_near_equal(prob['mixed.x3'], 0.0, 1e-6)
        assert_near_equal(prob['mixed.x45'], np.zeros((2, )), 1e-6)

        # Normally takes about 13 iters, but takes around 4 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 6)

    def test_simple_sellar_jacobian(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.linear_solver = om.DirectSolver(assemble_jac=False)

        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 4 iters, but takes around 3 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 4)

    def test_simple_sellar_jacobian_assembled(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 4 iters, but takes around 3 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 4)

    def test_simple_sellar_jacobian_assembled_dense(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.options['assembled_jac_type'] = 'dense'
        model.nonlinear_solver.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 4 iters, but takes around 3 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 4)

    def test_simple_sellar_full(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.linear_solver = om.DirectSolver()
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 5 iters, but takes around 4 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 6)

    def test_simple_sellar_full_jacobian(self):
        # Test top level Sellar (i.e., not grouped).

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.linear_solver = om.DirectSolver()

        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 5 iters, but takes around 4 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 5)

    def test_jacobian_update_converge_limit(self):
        # This model needs jacobian updates to converge.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.array([0, 20.0])))
        model.add_subsystem('comp', SpedicatoHuang())

        model.connect('p1.x', 'comp.x')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['comp.y']
        model.nonlinear_solver.options['maxiter'] = 20
        model.nonlinear_solver.options['max_converge_failures'] = 1
        model.nonlinear_solver.options['diverge_limit'] = np.inf
        model.nonlinear_solver.linear_solver = om.DirectSolver()

        prob.setup()

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_near_equal(prob['comp.y'], np.array([-36.26230985,  10.20857237, -54.17658612]), 1e-6)

    def test_jacobian_update_diverge_limit(self):
        # This model needs jacobian updates to converge.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.array([0, 20.0])))
        model.add_subsystem('comp', SpedicatoHuang())

        model.connect('p1.x', 'comp.x')

        model.nonlinear_solver = om.BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['comp.y']
        model.nonlinear_solver.options['maxiter'] = 20
        model.nonlinear_solver.options['diverge_limit'] = 0.5
        model.nonlinear_solver.linear_solver = om.DirectSolver()

        prob.setup()

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_near_equal(prob['comp.y'], np.array([-36.26230985,  10.20857237, -54.17658612]), 1e-6)

    def test_backtracking(self):
        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = om.BroydenSolver()
        top.model.nonlinear_solver.options['maxiter'] = 25
        top.model.nonlinear_solver.options['diverge_limit'] = 0.5
        top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']

        top.model.linear_solver = om.DirectSolver()

        top.setup()

        # Setup again because we assigned a new linesearch
        top.setup()

        top.set_solver_print(level=2)
        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_near_equal(top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_near_equal(top['comp.z'], 2.5, 1e-8)

    def test_cs_around_broyden(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'),
                            promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'),
                            promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.BroydenSolver()
        sub.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=2)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

    def test_cs_around_broyden_compute_jac(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.BroydenSolver()
        sub.linear_solver = om.DirectSolver(assemble_jac=False)
        model.linear_solver = om.DirectSolver(assemble_jac=False)

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        sub.nonlinear_solver.options['compute_jacobian'] = True

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

    def test_cs_around_broyden_compute_jac_dense(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.BroydenSolver()
        sub.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        sub.nonlinear_solver.options['compute_jacobian'] = True

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

    def test_complex_step(self):
        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.ParallelGroup(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'),
                            promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'),
                            promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.BroydenSolver()
        sub.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-7)


# Commented the following test out until we fix the broyden check
# @unittest.skipUnless(MPI and PETScVector, "only run with MPI and PETSc.")
# class TestBryodenMPI(unittest.TestCase):

#     N_PROCS = 2

#     def test_distributed_comp(self):
#         prob = om.Problem()
#         model = prob.model
#         sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

#         sub.add_subsystem('d1', DistribExecComp(['y1 = 28 - 0.2*y2', 'y1 = 18 - 0.2*y2'], arr_size=2),
#                           promotes=['y1', 'y2'])
#         sub.add_subsystem('d2', DistribExecComp(['y2 = y1**.5 + 7', 'y2 = y1**.5 - 3'], arr_size=2),
#                           promotes=['y1', 'y2'])

#         sub.nonlinear_solver = om.BroydenSolver()
#         sub.linear_solver = om.LinearBlockGS()
#         model.linear_solver = om.LinearBlockGS()

#         prob.setup(check=False, force_alloc_complex=True)

#         with self.assertRaises(Exception) as cm:
#             prob.run_model()

#         msg = "BroydenSolver linear solver in Group (sub) cannot be used in or above a ParallelGroup or a " + \
#             "distributed component."
#         self.assertEqual(str(cm.exception), msg)


class TestBryodenFeature(unittest.TestCase):

    def test_sellar(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarStateConnection

        prob = om.Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
                                                   linear_solver=om.LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['state_eq.y2_command'], 12.05848819, .00001)

    def test_circuit(self):
        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = om.Problem()
        model = p.model

        model.add_subsystem('circuit', Circuit(), promotes_inputs=[('Vg', 'V'), ('I_in', 'I')])
        model.set_input_defaults('V', 0., units='V')
        model.set_input_defaults('I', 0.1, units='A')

        p.setup()

        # Replace existing solver with om.BroydenSolver
        model.circuit.nonlinear_solver = om.BroydenSolver()
        model.circuit.nonlinear_solver.options['maxiter'] = 20

        # Specify states for Broyden to solve
        model.circuit.nonlinear_solver.options['state_vars'] = ['n1.V', 'n2.V']

        model.nonlinear_solver.linear_solver = om.LinearBlockGS()

        # set some initial guesses
        p.set_val('circuit.n1.V', 10.)
        p.set_val('circuit.n2.V', 1.)

        p.set_solver_print(level=2)
        p.run_model()

        assert_near_equal(p.get_val('circuit.n1.V'), 9.90804735, 1e-5)
        assert_near_equal(p.get_val('circuit.n2.V'), 0.71278226, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_near_equal(p.get_val('circuit.R1.I') + p.get_val('circuit.D1.I'), .1, 1e-6)

    def test_circuit_options(self):
        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = om.Problem()
        model = p.model

        model.add_subsystem('circuit', Circuit(), promotes_inputs=[('Vg', 'V'), ('I_in', 'I')])
        model.set_input_defaults('V', 0., units='V')
        model.set_input_defaults('I', 0.1, units='A')

        p.setup()

        # Replace existing solver with BroydenSolver
        model.circuit.nonlinear_solver = om.BroydenSolver()
        model.circuit.nonlinear_solver.options['maxiter'] = 20
        model.circuit.nonlinear_solver.options['converge_limit'] = 0.1
        model.circuit.nonlinear_solver.options['max_converge_failures'] = 1

        # Specify states for Broyden to solve
        model.circuit.nonlinear_solver.options['state_vars'] = ['n1.V', 'n2.V']

        # set some initial guesses
        p.set_val('circuit.n1.V', 10.)
        p.set_val('circuit.n2.V', 1.)

        p.set_solver_print(level=2)
        p.run_model()

        assert_near_equal(p.get_val('circuit.n1.V'), 9.90804735, 1e-5)
        assert_near_equal(p.get_val('circuit.n2.V'), 0.71278226, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_near_equal(p.get_val('circuit.R1.I') + p.get_val('circuit.D1.I'), .1, 1e-6)

    def test_circuit_full(self):
        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = om.Problem()
        model = p.model

        model.add_subsystem('circuit', Circuit(), promotes_inputs=[('Vg', 'V'), ('I_in', 'I')])
        model.set_input_defaults('V', 0., units='V')
        model.set_input_defaults('I', 0.1, units='A')

        p.setup()

        # Replace existing solver with BroydenSolver
        model.circuit.nonlinear_solver = om.BroydenSolver()
        model.circuit.nonlinear_solver.options['maxiter'] = 20
        model.circuit.nonlinear_solver.linear_solver = om.DirectSolver()

        # set some initial guesses
        p.set_val('circuit.n1.V', 10.)
        p.set_val('circuit.n2.V', 1.)

        p.set_solver_print(level=2)
        p.run_model()

        assert_near_equal(p.get_val('circuit.n1.V'), 9.90804735, 1e-5)
        assert_near_equal(p.get_val('circuit.n2.V'), 0.71278226, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_near_equal(p.get_val('circuit.R1.I') + p.get_val('circuit.D1.I'), .1, 1e-6)


if __name__ == "__main__":
    unittest.main()
