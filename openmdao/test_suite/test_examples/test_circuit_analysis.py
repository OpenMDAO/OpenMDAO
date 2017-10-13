from __future__ import print_function, division, absolute_import
import numpy as np

import unittest

from openmdao.api import ExplicitComponent, ImplicitComponent

from openmdao.devtools.testutil import assert_rel_error


class Resistor(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('R', default=1., desc='Resistance in Ohms')

    def setup(self):
        self.add_input('V_in', units='V')
        self.add_input('V_out', units='V')
        self.add_output('I', units='A')

        self.declare_partials('I', 'V_in', method='fd')
        self.declare_partials('I', 'V_out', method='fd')

    def compute(self, i, o):
        deltaV = i['V_in'] - i['V_out']
        o['I'] = deltaV / self.metadata['R']


class Diode(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('Is', default=1e-15, desc='Saturation current in Amps')
        self.metadata.declare('Vt', default=.025875, desc='Thermal voltage in Volts')

    def setup(self):
        self.add_input('V_in', units='V')
        self.add_input('V_out', units='V')
        self.add_output('I', units='A')

        self.declare_partials('I', 'V_in', method='fd')
        self.declare_partials('I', 'V_out', method='fd')

    def compute(self, i, o):
        deltaV = i['V_in'] - i['V_out']
        Is = self.metadata['Is']
        Vt = self.metadata['Vt']
        o['I'] = Is * np.exp(deltaV / Vt - 1)


class Node(ImplicitComponent):
    def initialize(self):
        self.metadata.declare('n_in', default=1, type_=int, desc='number of connections with + assumed in')
        self.metadata.declare('n_out', default=1, type_=int, desc='number of current connections + assumed out')

    def setup(self):
        self.add_output('V', val=5., units='V')

        for i in range(self.metadata['n_in']):
            i_name = 'I_in:{}'.format(i)
            self.add_input(i_name, units='A')

        for i in range(self.metadata['n_out']):
            i_name = 'I_out:{}'.format(i)
            self.add_input(i_name, units='A')

        #note: we don't declare any partials wrt `V` here,
        #      because the residual doesn't directly depend on it
        self.declare_partials('V', 'I*', method='fd')

    def apply_nonlinear(self, i, o, r):
        r['V'] = 0.
        for i_conn in range(self.metadata['n_in']):
            r['V'] += i['I_in:{}'.format(i_conn)]
        for i_conn in range(self.metadata['n_out']):
            r['V'] -= i['I_out:{}'.format(i_conn)]

class TestCircuit(unittest.TestCase):

    def test_circuit(self):


        from openmdao.api import Group, NewtonSolver, DirectSolver, \
            ArmijoGoldsteinLS, Problem, IndepVarComp

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
                self.nonlinear_solver.options['maxiter'] = 1000
                self.nonlinear_solver.options['solve_subsystems'] = True
                self.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
                self.nonlinear_solver.linesearch.options['maxiter'] = 10
                self.nonlinear_solver.linesearch.options['iprint'] = 2
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
        p['circuit.n2.V'] = 1e-3

        p.run_model()

        assert_rel_error(self, p['circuit.n1.V'], 9.90830282, 1e-5)
        assert_rel_error(self, p['circuit.n2.V'], 0.73858486, 1e-5)
        assert_rel_error(self, p['circuit.R1.I'], 0.09908303, 1e-5)
        assert_rel_error(self, p['circuit.R2.I'], 0.00091697, 1e-5)
        assert_rel_error(self, p['circuit.D1.I'], 0.00091697, 1e-5)
        #'Sanity check: shoudl sum to .1 Amps
        assert_rel_error(self,  p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)


if __name__ == "__main__":
    unittest.main()