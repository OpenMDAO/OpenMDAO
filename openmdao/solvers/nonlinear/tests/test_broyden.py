"""Test the Broyden nonlinear solver. """
from __future__ import print_function

import unittest
import warnings

import numpy as np

from openmdao.api import Problem, LinearRunOnce, ImplicitComponent, IndepVarComp, DirectSolver, \
     BoundsEnforceLS, LinearBlockGS
from openmdao.solvers.nonlinear.broyden import BroydenSolver
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.components.sellar import SellarStateConnection, SellarDerivatives
from openmdao.utils.assert_utils import assert_rel_error


class VectorEquation(ImplicitComponent):
    """Equation with 5 states in a single vector. Should converge to x=[0,0,0,0,0]"""

    def setup(self):
        self.add_input('c', 0.01)
        self.add_output('x', np.ones((5, )))

    def apply_nonlinear(self, inputs, outputs, residuals):
        c = inputs['c']
        x = outputs['x']

        d = np.array([3, 2, 1.5, 1, 0.5])

        residuals['x'] = -d*x - c*x**3


class MixedEquation(ImplicitComponent):
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


class SpedicatoHuang(ImplicitComponent):

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

    def test_error_badname(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        model.nonlinear_solver.options['state_vars'] = ['junk']

        with self.assertRaises(ValueError) as context:
            prob.run_model()

        msg = "The following variable names were not found: junk"
        self.assertEqual(str(context.exception), msg)

    def test_error_need_direct_solver(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        with self.assertRaises(ValueError) as context:
            prob.run_model()

        msg = "Linear solver must be DirectSolver when solving the full model."
        self.assertEqual(str(context.exception), msg)

    def test_simple_sellar(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

    def test_simple_sellar_cycle(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarDerivatives(nonlinear_solver=BroydenSolver(),
                                               linear_solver=LinearRunOnce())

        prob.setup(check=False)

        model.nonlinear_solver.options['state_vars'] = ['y1']
        model.nonlinear_solver.options['compute_jacobian'] = True

        prob.set_solver_print(level=2)

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_sellar_state_connection_fd_system(self):
        # Sellar model closes loop with state connection instead of a cycle.
        # This test is just fd.
        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())
        prob.model.approx_totals(method='fd')

        prob.setup(check=False)

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 6)

    def test_vector(self):
        # Testing Broyden on a 5 state single vector case.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('vec', VectorEquation())

        model.connect('p1.c', 'vec.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['vec.x']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['vec.x'], np.zeros((5, )), 1e-6)

    def test_mixed(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12', 'mixed.x3', 'mixed.x45']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['mixed.x12'], np.zeros((2, )), 1e-6)
        assert_rel_error(self, prob['mixed.x3'], 0.0, 1e-6)
        assert_rel_error(self, prob['mixed.x45'], np.zeros((2, )), 1e-6)

    def test_missing_state_warning(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup(check=False)

        with warnings.catch_warnings(record=True) as w:
            prob.run_model()

        self.assertEqual(len(w), 1)

        msg = "The following states are not covered by a solver, and may have been omitted from the BroydenSolver 'state_vars': mixed.x3, mixed.x45"

        self.assertEqual(str(w[0].message), msg)

        # Try again with promoted names.
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation(), promotes=['*'])

        model.connect('p1.c', 'c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['x12']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup(check=False)

        with warnings.catch_warnings(record=True) as w:
            prob.run_model()

        self.assertEqual(len(w), 1)

        msg = "The following states are not covered by a solver, and may have been omitted from the BroydenSolver 'state_vars': x3, x45"

        self.assertEqual(str(w[0].message), msg)

    def test_mixed_promoted_vars(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation(), promotes_outputs=['x12', 'x3', 'x45'])

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['x12', 'x3', 'x45']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['x12'], np.zeros((2, )), 1e-6)
        assert_rel_error(self, prob['x3'], 0.0, 1e-6)
        assert_rel_error(self, prob['x45'], np.zeros((2, )), 1e-6)

    def test_mixed_jacobian(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12', 'mixed.x3', 'mixed.x45']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['mixed.x12'], np.zeros((2, )), 1e-6)
        assert_rel_error(self, prob['mixed.x3'], 0.0, 1e-6)
        assert_rel_error(self, prob['mixed.x45'], np.zeros((2, )), 1e-6)

        # Normally takes about 13 iters, but takes around 4 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 6)

    def test_simple_sellar_jacobian(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 4 iters, but takes around 3 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 4)

    def test_simple_sellar_full(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        model.nonlinear_solver.linear_solver = DirectSolver()
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 5 iters, but takes around 4 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 6)

    def test_simple_sellar_full_jacobian(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 5 iters, but takes around 4 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 5)

    def test_jacobian_update_converge_limit(self):
        # This model needs jacobian updates to converge.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', np.array([0, 20.0])))
        model.add_subsystem('comp', SpedicatoHuang())

        model.connect('p1.x', 'comp.x')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['comp.y']
        model.nonlinear_solver.options['maxiter'] = 20
        model.nonlinear_solver.options['max_converge_failures'] = 1
        model.nonlinear_solver.options['diverge_limit'] = np.inf
        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.setup(check=False)

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], np.array([-36.26230985,  10.20857237, -54.17658612]), 1e-6)

    def test_jacobian_update_diverge_limit(self):
        # This model needs jacobian updates to converge.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', np.array([0, 20.0])))
        model.add_subsystem('comp', SpedicatoHuang())

        model.connect('p1.x', 'comp.x')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['comp.y']
        model.nonlinear_solver.options['maxiter'] = 20
        model.nonlinear_solver.options['diverge_limit'] = 0.5
        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.setup(check=False)

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], np.array([-36.26230985,  10.20857237, -54.17658612]), 1e-6)

    def test_backtracking(self):
        top = Problem()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = BroydenSolver()
        top.model.nonlinear_solver.options['maxiter'] = 25
        top.model.nonlinear_solver.options['diverge_limit'] = 0.5
        top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']

        top.model.linear_solver = DirectSolver()

        top.setup(check=False)
        top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='vector')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        top.set_solver_print(level=2)
        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_rel_error(self, top['comp.z'], 2.5, 1e-8)


class TestBryodenFeature(unittest.TestCase):

    def test_sellar(self):
        from openmdao.api import Problem, LinearRunOnce, IndepVarComp, BroydenSolver
        from openmdao.test_suite.components.sellar import SellarStateConnection

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup()

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.options['compute_jacobian'] = False

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

    def test_circuit(self):
        from openmdao.api import Group, BroydenSolver, DirectSolver, Problem, IndepVarComp

        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # Replace existing solver with BroydenSolver
        model.circuit.nonlinear_solver = BroydenSolver()
        model.circuit.nonlinear_solver.options['maxiter'] = 20

        # Specify states for Broyden to solve
        model.circuit.nonlinear_solver.options['state_vars'] = ['n1.V', 'n2.V']

        model.nonlinear_solver.linear_solver = LinearBlockGS()

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1.

        p.set_solver_print(level=2)
        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.71278185, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_rel_error(self,  p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_options(self):
        from openmdao.api import Group, BroydenSolver, DirectSolver, Problem, IndepVarComp

        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # Replace existing solver with BroydenSolver
        model.circuit.nonlinear_solver = BroydenSolver()
        model.circuit.nonlinear_solver.options['maxiter'] = 20
        model.circuit.nonlinear_solver.options['converge_limit'] = 0.1
        model.circuit.nonlinear_solver.options['max_converge_failures'] = 1

        # Specify states for Broyden to solve
        model.circuit.nonlinear_solver.options['state_vars'] = ['n1.V', 'n2.V']

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1.

        p.set_solver_print(level=2)
        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.71278185, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_rel_error(self,  p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_full(self):
        from openmdao.api import Group, BroydenSolver, DirectSolver, Problem, IndepVarComp

        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # Replace existing solver with BroydenSolver
        model.circuit.nonlinear_solver = BroydenSolver()
        model.circuit.nonlinear_solver.options['maxiter'] = 20
        model.circuit.nonlinear_solver.linear_solver = DirectSolver()

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1.

        p.set_solver_print(level=2)
        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.71278185, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_rel_error(self,  p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

if __name__ == "__main__":
    unittest.main()