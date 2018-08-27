""" Test for the relaxation Line Search"""

import unittest

from openmdao.api import Problem, IndepVarComp, ScipyKrylov, NewtonSolver
from openmdao.solvers.linesearch.relaxation import RelaxationLS
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit
from openmdao.utils.assert_utils import assert_rel_error


class TestRelaxationLS(unittest.TestCase):

    def test_bad_settings(self):
        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        newton = p.model.circuit.nonlinear_solver
        newton.linesearch = RelaxationLS()
        newton.linesearch.options['relax_far'] = 1.0
        newton.linesearch.options['relax_near'] = 2.0

        with self.assertRaises(Exception) as raises_cm:
            p.final_setup()

        exception = raises_cm.exception
        msg = "In options, relax_far must be greater than or equal to relax_near."

        self.assertEqual(exception.args[0], msg)

    def test_circuit_advanced_newton(self):
        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # you can change the NewtonSolver settings in circuit after setup is called
        newton = p.model.circuit.nonlinear_solver
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.linesearch = RelaxationLS()
        newton.linesearch.options['initial_relaxation'] = 0.019
        newton.linesearch.options['relax_far'] = 1.15e-3
        newton.linesearch.options['relax_near'] = 1.13e-3

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        p.run_model()

    def test_linesearch_bounds_vector(self):
        top = Problem()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = RelaxationLS(bound_enforcement='vector')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

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

    def test_linesearch_bounds_wall(self):
        top = Problem()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = RelaxationLS(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

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

    def test_linesearch_bounds_scalar(self):
        top = Problem()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = RelaxationLS(bound_enforcement='scalar')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bound: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        self.assertTrue(1.5 <= top['comp.z'] <= 1.6)

        # Test lower bound: should stop just short of the upper bound
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        self.assertTrue(2.4 <= top['comp.z'] <= 2.5)


class TestFeatureRelaxationLS(unittest.TestCase):

    def test_circuit_advanced_newton(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.solvers.linesearch.relaxation import RelaxationLS

        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # You can change the NewtonSolver settings in circuit after setup is called
        newton = p.model.circuit.nonlinear_solver
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True

        # Tailored the relaxation settings to give good convergence.
        newton.linesearch = RelaxationLS()
        newton.linesearch.options['initial_relaxation'] = 0.025
        newton.linesearch.options['relax_far'] = 1.e-3
        newton.linesearch.options['relax_near'] = 1.e-5

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        p.run_model()

        assert_rel_error(self, p['circuit.R1.I'], 0.09908047, 1e-5)
        assert_rel_error(self, p['circuit.R2.I'], 0.00091953, 1e-5)
        assert_rel_error(self, p['circuit.D1.I'], 0.00091953, 1e-5)

if __name__ == "__main__":
    unittest.main()
