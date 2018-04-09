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

model.add_design_var('ground.V')
model.add_design_var('source.I')
model.add_objective('circuit.D1.I')

p.setup(check=False)

# set some initial guesses
p['circuit.n1.V'] = 10.
p['circuit.n2.V'] = 1.

p.run_model()
