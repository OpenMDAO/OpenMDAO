"""Test the Broyden nonlinear solver. """

import unittest

import numpy as np

import openmdao.api as om

from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.components.sellar import SellarStateConnection, SellarDerivatives, \
     SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.scripts.circuit_analysis import Circuit
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_totals

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


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
        x[2] = outputs['x3'].item()
        x[3:] = outputs['x45']

        d = np.array([3, 2, 1.5, 1, 0.5])

        res = -d*x - c*x**3
        residuals['x12'] = res[:2]
        residuals['x3'] = res[2]
        residuals['x45'] = res[3:]

    def linearize(self, inputs, outputs, jacobian):
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

        msg = "Solver 'NL: BROYDEN' on system 'g1' failed to converge in 1 iterations."
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
        prob.model = SellarStateConnection(nonlinear_solver=om.BroydenSolver(),
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
        assert_check_totals(totals)

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
        assert_check_totals(totals)

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
        assert_check_totals(totals)

    def test_complex_step(self):
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
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(totals)


# Commented the following test out until we fix the broyden check
# @unittest.skipUnless(MPI and PETScVector, "only run with MPI and PETSc.")
# class TestBryodenMPI(unittest.TestCase):

#     N_PROCS = 2

#     def test_distributed_comp(self):
#         class Y1Comp(om.ExplicitComponent):
#             def __init__(self, arr_size=11, **kwargs):
#                 super().__init__(**kwargs)
#                 self.arr_size = arr_size
#                 self.options['distributed'] = True

#             def setup(self):
#                 comm = self.comm
#                 rank = comm.rank

#                 sizes, _ = evenly_distrib_idxs(comm.size, self.arr_size)

#                 self.add_input('y2', np.ones(sizes[rank]))
#                 self.add_output('y1', np.ones(sizes[rank]))

#                 self.declare_partials(of='y1', wrt='y2', method='cs')

#             def compute(self, inputs, outputs):
#                 if self.comm.rank == 0:
#                     outputs['y1'] = 28. - .2 * inputs['y2']
#                 else:
#                     outputs['y1'] = 18. - .2 * inputs['y2']

#         class Y2Comp(om.ExplicitComponent):
#             def __init__(self, arr_size=11, **kwargs):
#                 super().__init__(**kwargs)
#                 self.arr_size = arr_size
#                 self.options['distributed'] = True

#             def setup(self):
#                 comm = self.comm
#                 rank = comm.rank

#                 sizes, _ = evenly_distrib_idxs(comm.size, self.arr_size)

#                 self.add_input('y1', np.ones(sizes[rank]))
#                 self.add_output('y2', np.ones(sizes[rank]))

#                 self.declare_partials(of='y2', wrt='y1', method='cs')

#             def compute(self, inputs, outputs):
#                 if self.comm.rank == 0:
#                     outputs['y2'] = inputs['y1'] ** .5 + 7.
#                 else:
#                     outputs['y2'] = inputs['y1'] ** .5 - 3.

#         prob = om.Problem()
#         model = prob.model
#         sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

#         sub.add_subsystem('d1', Y1Comp(arr_size=2), promotes=['y1', 'y2'])
#         sub.add_subsystem('d2', Y2Comp(arr_size=2), promotes=['y1', 'y2'])

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

    def test_circuit(self):

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


class _CSSquare(om.ExplicitComponent):
    """y = x**2, scalar or elementwise, with analytic partials so it survives an outer cs."""

    def initialize(self):
        self.options.declare('size', types=int, default=1)

    def setup(self):
        size = self.options['size']
        if size == 1:
            self.add_input('x', val=1.0)
            self.add_output('y', val=1.0)
            self.declare_partials('y', 'x')
        else:
            self.add_input('x', val=np.ones(size))
            self.add_output('y', val=np.ones(size))
            self.declare_partials('y', 'x', rows=np.arange(size), cols=np.arange(size))

    def compute(self, inputs, outputs):
        outputs['y'] = inputs['x'] ** 2

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = 2.0 * inputs['x']


class _CSRoot(om.ImplicitComponent):
    """R(s) = s**2 - a, so s = sqrt(a) and ds/da = 1/(2*sqrt(a))."""

    def setup(self):
        self.add_input('a', val=4.0)
        self.add_output('s', val=1.5, lower=0.0, upper=10.0)
        self.declare_partials('s', ['a', 's'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['s'] = outputs['s'] ** 2 - inputs['a']

    def linearize(self, inputs, outputs, partials):
        partials['s', 's'] = 2.0 * outputs['s']
        partials['s', 'a'] = -1.0


class _CSCycleA(om.ExplicitComponent):
    """y1 = a - k*y2**2, nonlinear in the coupling."""

    def initialize(self):
        self.options.declare('k', default=0.1)

    def setup(self):
        self.add_input('a', val=3.0)
        self.add_input('y2', val=0.0)
        self.add_output('y1', val=1.0)
        self.declare_partials('y1', ['a', 'y2'])

    def compute(self, inputs, outputs):
        outputs['y1'] = inputs['a'] - self.options['k'] * inputs['y2'] ** 2

    def compute_partials(self, inputs, partials):
        partials['y1', 'a'] = 1.0
        partials['y1', 'y2'] = -2.0 * self.options['k'] * inputs['y2']


class _CSCycleB(om.ExplicitComponent):
    """y2 = k*y1."""

    def initialize(self):
        self.options.declare('k', default=0.1)

    def setup(self):
        self.add_input('y1', val=1.0)
        self.add_output('y2', val=1.0)
        self.declare_partials('y2', 'y1')

    def compute(self, inputs, outputs):
        outputs['y2'] = self.options['k'] * inputs['y1']

    def compute_partials(self, inputs, partials):
        partials['y2', 'y1'] = self.options['k']


class TestBroydenCSIndepVarNudge(unittest.TestCase):
    """Independent outputs must not be translated by the cs_reconverge nudge. Issue #3810."""

    def _feedforward(self, irrelevant=None, state_vars=None, size=1):
        prob = om.Problem()
        model = prob.model
        ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        if size == 1:
            ivc.add_output('x', 22.0)
        else:
            ivc.add_output('x', np.array([3.0, 22.0, 5.0]))
        if irrelevant is not None:
            ivc.add_output('irrelevant', irrelevant)
        model.add_subsystem('sq', _CSSquare(size=size), promotes=['*'])

        solver = model.nonlinear_solver = om.BroydenSolver()
        solver.options['cs_reconverge'] = True
        if state_vars is not None:
            solver.options['state_vars'] = state_vars
        solver.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()
        return prob

    def _cs_total(self, prob, of='y', wrt='x'):
        data = prob.check_totals(of=[of], wrt=[wrt], method='cs', out_stream=None)
        return data[of, wrt]['J_fd']

    def _run(self, prob):
        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.run_model()
        return prob

    def test_indep_var_not_shifted_full_inverse(self):
        # The issue #3810 reproducer. In full inverse mode the Broyden state vector is the
        # whole output vector, so this cannot be fixed by looking at Broyden's state alone.
        prob = self._run(self._feedforward())
        assert_near_equal(self._cs_total(prob).item(), 44.0, 1e-13)

    def test_indep_var_not_shifted_explicit_state_vars(self):
        # Same defect reached through the other Broyden mode. Here the nudge moves an output
        # that Broyden does not own and never writes back.
        prob = self._run(self._feedforward(state_vars=['y']))
        assert_near_equal(self._cs_total(prob).item(), 44.0, 1e-13)

    def test_disconnected_output_does_not_change_derivative(self):
        # The magnitude of a completely disconnected output must not reach the derivative
        # through the norm that sizes the nudge.
        results = [self._cs_total(self._run(self._feedforward(irrelevant=v))).item()
                   for v in (0.0, 1.0, 1e8, 1e10)]
        for value in results:
            assert_near_equal(value, 44.0, 1e-13)
        self.assertEqual(len(set(results)), 1,
                         msg='a disconnected output changed the reported derivative')

    def test_disconnected_output_does_not_change_iteration_count(self):
        # Restoring independent values after a full nudge would still leave the sizing norm
        # contaminated, which steers how hard the solver works. Pin the behavior, not just
        # the number.
        counts = []
        for value in (0.0, 1e6, 1e8, 1e10):
            prob = self._run(self._nonlinear_cycle(irrelevant=value))
            self._cs_total(prob, of='y1', wrt='a')
            counts.append(prob.model.nonlinear_solver._iter_count)
        self.assertEqual(len(set(counts)), 1,
                         msg=f'disconnected output changed Broyden iteration count: {counts}')

    def test_auto_ivc_indep_var(self):
        # _auto_ivc outputs carry the same tag and must be protected the same way.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('sq', _CSSquare(), promotes=['*'])
        solver = model.nonlinear_solver = om.BroydenSolver()
        solver.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()
        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.set_val('x', 22.0)
        prob.run_model()
        assert_near_equal(self._cs_total(prob).item(), 44.0, 1e-13)

    def test_vector_indep_var(self):
        # Every element of a vector independent variable must be protected, not just the
        # first. A scalar-only mask passes the scalar tests and fails here.
        prob = self._run(self._feedforward(irrelevant=1e8, size=3))
        x = np.array([3.0, 22.0, 5.0])
        expected = np.diag([2.0 * x[0], 2.0 * x[1], 2.0 * x[2]])
        assert_near_equal(self._cs_total(prob), expected, 1e-12)

    def test_indep_var_declared_as_state_var(self):
        # An independent output may legally be named in state_vars. Excluding it by
        # Broyden's state membership rather than by tag reintroduces the defect here.
        prob = self._run(self._feedforward(irrelevant=1e8, state_vars=['x', 'y']))
        assert_near_equal(self._cs_total(prob).item(), 44.0, 1e-13)

    def _nonlinear_cycle(self, irrelevant=None, state_vars=('y1', 'y2')):
        prob = om.Problem()
        model = prob.model
        ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('a', 3.0)
        if irrelevant is not None:
            ivc.add_output('irrelevant', irrelevant)
        model.add_subsystem('c1', _CSCycleA(k=0.1), promotes=['*'])
        model.add_subsystem('c2', _CSCycleB(k=0.1), promotes=['*'])
        solver = model.nonlinear_solver = om.BroydenSolver()
        solver.options['maxiter'] = 60
        solver.options['atol'] = 1e-14
        solver.options['rtol'] = 1e-14
        if state_vars is not None:
            solver.options['state_vars'] = list(state_vars)
        solver.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()
        return prob

    def test_coupled_states_still_reconverge(self):
        # The nudge is retained for genuine solver states. Removing it entirely leaves this
        # cycle unconverged under complex step.
        prob = self._run(self._nonlinear_cycle(irrelevant=1e8))
        expected = 1.0 / np.sqrt(1.0 + 4.0 * 0.1 ** 3 * 3.0)
        assert_near_equal(self._cs_total(prob, of='y1', wrt='a').item(), expected, 1e-10)

    def test_implicit_state_derivative(self):
        # A real implicit state, where the wrong base point shows up as a wrong analytic
        # value rather than a convergence artifact.
        prob = om.Problem()
        model = prob.model
        ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('a', 4.0)
        ivc.add_output('irrelevant', 1e8)
        model.add_subsystem('root', _CSRoot(), promotes=['*'])
        solver = model.nonlinear_solver = om.BroydenSolver()
        solver.options['state_vars'] = ['s']
        solver.options['maxiter'] = 60
        solver.options['atol'] = 1e-14
        solver.options['rtol'] = 1e-14
        solver.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()
        self._run(prob)
        assert_near_equal(self._cs_total(prob, of='s', wrt='a').item(), 0.25, 1e-10)

    def test_with_linesearch(self):
        # The nudge runs before the linesearch ever sees the vector. Bounds enforcement
        # must not reintroduce a shift on independent outputs.
        prob = om.Problem()
        model = prob.model
        ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('a', 4.0)
        ivc.add_output('irrelevant', 1e8)
        model.add_subsystem('root', _CSRoot(), promotes=['*'])
        solver = model.nonlinear_solver = om.BroydenSolver()
        solver.options['state_vars'] = ['s']
        solver.options['maxiter'] = 60
        solver.options['atol'] = 1e-14
        solver.options['rtol'] = 1e-14
        solver.linesearch = om.BoundsEnforceLS()
        solver.linear_solver = om.DirectSolver()
        model.linear_solver = om.DirectSolver()
        self._run(prob)
        assert_near_equal(self._cs_total(prob, of='s', wrt='a').item(), 0.25, 1e-10)

    def test_repeated_evaluations_and_fd(self):
        # Repeated complex steps must be idempotent, and fd must be unaffected because the
        # modified block is gated on complex step.
        prob = self._run(self._feedforward(irrelevant=1e8))
        first = self._cs_total(prob).item()
        second = self._cs_total(prob).item()
        third = self._cs_total(prob).item()
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        assert_near_equal(first, 44.0, 1e-13)

        other = self._run(self._feedforward(irrelevant=1e8))
        fd = other.check_totals(of=['y'], wrt=['x'], method='fd', out_stream=None)
        assert_near_equal(fd['y', 'x']['J_fd'].item(), 44.0, 1e-5)

    def test_cs_reconverge_nudge_semantics(self):
        # White-box check of the nudge itself: it must leave every imaginary part untouched,
        # leave outputs tagged 'openmdao:indep_var' untouched, and shift the remaining
        # outputs by exactly norm(those outputs) * 1e-10.
        prob = self._run(self._feedforward(irrelevant=1e8))
        model = prob.model
        solver = model.nonlinear_solver
        model._set_complex_step_mode(True)
        try:
            arr = model._outputs.asarray()
            idx = {name: model._outputs.get_range(name)[0]
                   for name in model._outputs._abs_iter()}
            arr[idx['ivc.x']] += 1e-40j
            arr[idx['sq.y']] += 3e-41j
            before = arr.copy()
            expected_nudge = np.linalg.norm(before[[idx['sq.y']]]) * 1e-10

            captured = {}
            original_get_vector = solver.get_vector

            def capture(vec):
                if 'arr' not in captured:
                    captured['arr'] = model._outputs.asarray().copy()
                return original_get_vector(vec)

            solver.get_vector = capture
            try:
                solver._iter_initialize()
            finally:
                solver.get_vector = original_get_vector

            # the first get_vector call is the statement right after the nudge
            after = captured['arr']
            self.assertTrue(np.array_equal(after.imag, before.imag),
                            msg='nudge must not modify imaginary parts')
            self.assertEqual((after[idx['ivc.x']] - before[idx['ivc.x']]).real, 0.0,
                             msg='indep var was translated by the nudge')
            self.assertEqual((after[idx['ivc.irrelevant']] -
                              before[idx['ivc.irrelevant']]).real, 0.0,
                             msg='disconnected indep var was translated by the nudge')
            # exact, including the single float64 rounding of (y + nudge)
            base = before[idx['sq.y']].real
            expected_delta = (base + expected_nudge) - base
            delta = (after[idx['sq.y']] - before[idx['sq.y']]).real
            self.assertEqual(delta, expected_delta,
                             msg='solver-state nudge missing or mis-sized')
        finally:
            model._set_complex_step_mode(False)

    def test_real_solve_and_fd_unchanged(self):
        # The modified block is gated on complex step, so a plain real solve must be
        # bit-identical to what it always was.
        prob = self._run(self._nonlinear_cycle(irrelevant=1e8))
        y1 = prob.get_val('y1')[0]
        expected = (-1.0 + np.sqrt(1.0 + 4.0 * 0.1 ** 3 * 3.0)) / (2.0 * 0.1 ** 3)
        assert_near_equal(y1, expected, 1e-10)
        self.assertFalse(np.iscomplexobj(prob.model._outputs.asarray()))


if __name__ == "__main__":
    unittest.main()
