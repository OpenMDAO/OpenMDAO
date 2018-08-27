from __future__ import print_function, division, absolute_import
import numpy as np

import unittest

from openmdao.api import ExplicitComponent, ImplicitComponent, Group, NewtonSolver, DirectSolver

from openmdao.utils.assert_utils import assert_rel_error


class Resistor(ExplicitComponent):
    """Computes current across a resistor using Ohm's law."""

    def initialize(self):
        self.options.declare('R', default=1., desc='Resistance in Ohms')

    def setup(self):
        self.add_input('V_in', units='V')
        self.add_input('V_out', units='V')
        self.add_output('I', units='A')

        self.declare_partials('I', 'V_in', method='fd')
        self.declare_partials('I', 'V_out', method='fd')

    def compute(self, inputs, outputs):
        deltaV = inputs['V_in'] - inputs['V_out']
        outputs['I'] = deltaV / self.options['R']


class Diode(ExplicitComponent):
    """Computes current across a diode using the Shockley diode equation."""

    def initialize(self):
        self.options.declare('Is', default=1e-15, desc='Saturation current in Amps')
        self.options.declare('Vt', default=.025875, desc='Thermal voltage in Volts')

    def setup(self):
        self.add_input('V_in', units='V')
        self.add_input('V_out', units='V')
        self.add_output('I', units='A')

        self.declare_partials('I', 'V_in', method='fd')
        self.declare_partials('I', 'V_out', method='fd')

    def compute(self, inputs, outputs):
        deltaV = inputs['V_in'] - inputs['V_out']
        Is = self.options['Is']
        Vt = self.options['Vt']
        outputs['I'] = Is * (np.exp(deltaV / Vt) - 1.0)


class Node(ImplicitComponent):
    """Computes voltage residual across a node based on incoming and outgoing current."""

    def initialize(self):
        self.options.declare('n_in', default=1, types=int, desc='number of connections with + assumed in')
        self.options.declare('n_out', default=1, types=int, desc='number of current connections + assumed out')

    def setup(self):
        self.add_output('V', val=5., units='V')

        for i in range(self.options['n_in']):
            i_name = 'I_in:{}'.format(i)
            self.add_input(i_name, units='A')

        for i in range(self.options['n_out']):
            i_name = 'I_out:{}'.format(i)
            self.add_input(i_name, units='A')

        #note: we don't declare any partials wrt `V` here,
        #      because the residual doesn't directly depend on it
        self.declare_partials('V', 'I*', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['V'] = 0.
        for i_conn in range(self.options['n_in']):
            residuals['V'] += inputs['I_in:{}'.format(i_conn)]
        for i_conn in range(self.options['n_out']):
            residuals['V'] -= inputs['I_out:{}'.format(i_conn)]


# note: This is defined twice in the file. Once so you can import it, and once inside a test that gets included in the docs.
class Circuit(Group):

    def setup(self):
        self.add_subsystem('n1', Node(n_in=1, n_out=2), promotes_inputs=[('I_in:0', 'I_in')])
        self.add_subsystem('n2', Node())  # leaving defaults

        self.add_subsystem('R1', Resistor(R=100.), promotes_inputs=[('V_out', 'Vg')])
        self.add_subsystem('R2', Resistor(R=10000.))
        self.add_subsystem('D1', Diode(), promotes_inputs=[('V_out', 'Vg')])

        self.connect('n1.V', ['R1.V_in', 'R2.V_in'])
        self.connect('R1.I', 'n1.I_out:0')
        self.connect('R2.I', 'n1.I_out:1')

        self.connect('n2.V', ['R2.V_out', 'D1.V_in'])
        self.connect('R2.I', 'n2.I_in:0')
        self.connect('D1.I', 'n2.I_out:0')

        self.nonlinear_solver = NewtonSolver()
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = DirectSolver()


class TestCircuit(unittest.TestCase):

    def test_circuit_plain_newton_assembled(self):

        from openmdao.api import Group, NewtonSolver, DirectSolver, Problem, IndepVarComp

        from openmdao.test_suite.test_examples.test_circuit_analysis import Resistor, Diode, Node

        class Circuit(Group):

            def setup(self):
                self.add_subsystem('n1', Node(n_in=1, n_out=2), promotes_inputs=[('I_in:0', 'I_in')])
                self.add_subsystem('n2', Node())  # leaving defaults

                self.add_subsystem('R1', Resistor(R=100.), promotes_inputs=[('V_out', 'Vg')])
                self.add_subsystem('R2', Resistor(R=10000.))
                self.add_subsystem('D1', Diode(), promotes_inputs=[('V_out', 'Vg')])

                self.connect('n1.V', ['R1.V_in', 'R2.V_in'])
                self.connect('R1.I', 'n1.I_out:0')
                self.connect('R2.I', 'n1.I_out:1')

                self.connect('n2.V', ['R2.V_out', 'D1.V_in'])
                self.connect('R2.I', 'n2.I_in:0')
                self.connect('D1.I', 'n2.I_out:0')

                self.nonlinear_solver = NewtonSolver()
                self.nonlinear_solver.options['iprint'] = 2
                self.nonlinear_solver.options['maxiter'] = 20
                ##################################################################
                # Assemble at the group level. Default assembled jac type is 'csc'
                ##################################################################
                self.options['assembled_jac_type'] = 'csc'
                self.linear_solver = DirectSolver(assemble_jac=True)

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1.

        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.71278185, 1e-5)
        assert_rel_error(self, p['circuit.R1.I'], 0.09908047, 1e-5)
        assert_rel_error(self, p['circuit.R2.I'], 0.00091953, 1e-5)
        assert_rel_error(self, p['circuit.D1.I'], 0.00091953, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_rel_error(self,  p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_plain_newton(self):

        from openmdao.api import Group, NewtonSolver, DirectSolver, Problem, IndepVarComp

        from openmdao.test_suite.test_examples.test_circuit_analysis import Resistor, Diode, Node

        class Circuit(Group):

            def setup(self):
                self.add_subsystem('n1', Node(n_in=1, n_out=2), promotes_inputs=[('I_in:0', 'I_in')])
                self.add_subsystem('n2', Node())  # leaving defaults

                self.add_subsystem('R1', Resistor(R=100.), promotes_inputs=[('V_out', 'Vg')])
                self.add_subsystem('R2', Resistor(R=10000.))
                self.add_subsystem('D1', Diode(), promotes_inputs=[('V_out', 'Vg')])

                self.connect('n1.V', ['R1.V_in', 'R2.V_in'])
                self.connect('R1.I', 'n1.I_out:0')
                self.connect('R2.I', 'n1.I_out:1')

                self.connect('n2.V', ['R2.V_out', 'D1.V_in'])
                self.connect('R2.I', 'n2.I_in:0')
                self.connect('D1.I', 'n2.I_out:0')

                self.nonlinear_solver = NewtonSolver()
                self.nonlinear_solver.options['iprint'] = 2
                self.nonlinear_solver.options['maxiter'] = 20
                self.linear_solver = DirectSolver()

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1.

        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.71278185, 1e-5)
        assert_rel_error(self, p['circuit.R1.I'], 0.09908047, 1e-5)
        assert_rel_error(self, p['circuit.R2.I'], 0.00091953, 1e-5)
        assert_rel_error(self, p['circuit.D1.I'], 0.00091953, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_rel_error(self,  p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_plain_newton_many_iter(self):

        from openmdao.api import Problem, IndepVarComp

        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

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
        newton.options['maxiter'] = 50

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.98744708, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 8.73215484, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_rel_error(self,  p['circuit.R1.I'] + p['circuit.D1.I'], 0.09987447, 1e-6)

    def test_circuit_advanced_newton(self):
        from openmdao.api import ArmijoGoldsteinLS, Problem, IndepVarComp

        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

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
        newton.linesearch = ArmijoGoldsteinLS()
        newton.linesearch.options['maxiter'] = 10
        newton.linesearch.options['iprint'] = 2

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.71278185, 1e-5)
        assert_rel_error(self, p['circuit.R1.I'], 0.09908047, 1e-5)
        assert_rel_error(self, p['circuit.R2.I'], 0.00091953, 1e-5)
        assert_rel_error(self, p['circuit.D1.I'], 0.00091953, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_rel_error(self, p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_voltage_source(self):
        from openmdao.api import ArmijoGoldsteinLS, Problem, IndepVarComp, BalanceComp, ExecComp
        from openmdao.api import NewtonSolver, DirectSolver, NonlinearRunOnce, LinearRunOnce

        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))

        # replacing the fixed current source with a BalanceComp to represent a fixed Voltage source
        # model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('batt', IndepVarComp('V', 1.5, units='V'))
        bal = model.add_subsystem('batt_balance', BalanceComp())
        bal.add_balance('I', units='A', eq_units='V')

        model.add_subsystem('circuit', Circuit())
        model.add_subsystem('batt_deltaV', ExecComp('dV = V1 - V2', V1={'units':'V'}, V2={'units':'V'}, dV={'units':'V'}))

        # current into the circuit is now the output state from the batt_balance comp
        model.connect('batt_balance.I', 'circuit.I_in')
        model.connect('ground.V', ['circuit.Vg','batt_deltaV.V2'])
        model.connect('circuit.n1.V', 'batt_deltaV.V1')

        # set the lhs and rhs for the battery residual
        model.connect('batt.V', 'batt_balance.rhs:I')
        model.connect('batt_deltaV.dV', 'batt_balance.lhs:I')

        p.setup()

        ###################
        # Solver Setup
        ###################

        # change the circuit solver to RunOnce because we're
        # going to converge at the top level of the model with newton instead
        p.model.circuit.nonlinear_solver = NonlinearRunOnce()
        p.model.circuit.linear_solver = LinearRunOnce()

        # Put Newton at the top so it can also converge the new BalanceComp residual
        newton = p.model.nonlinear_solver = NewtonSolver()
        p.model.linear_solver = DirectSolver()
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 20
        newton.options['solve_subsystems'] = True
        newton.linesearch = ArmijoGoldsteinLS()
        newton.linesearch.options['maxiter'] = 10
        newton.linesearch.options['iprint'] = 2

        # set initial guesses from the current source problem
        p['circuit.n1.V'] = 9.8
        p['circuit.n2.V'] = .7

        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 1.5, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.65113362, 1e-5)
        assert_rel_error(self, p['circuit.R1.I'], 0.015, 1e-5)
        assert_rel_error(self, p['circuit.R2.I'], 8.48866375e-05, 1e-5)
        assert_rel_error(self, p['circuit.D1.I'], 8.48866375e-05, 1e-5)


if __name__ == "__main__":
    unittest.main()
