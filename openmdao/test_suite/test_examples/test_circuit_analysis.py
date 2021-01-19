import numpy as np

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class TestCircuit(unittest.TestCase):

    def test_circuit_plain_newton_assembled(self):

        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Resistor, Diode, Node

        class Circuit(om.Group):

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

                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.nonlinear_solver.options['iprint'] = 2
                self.nonlinear_solver.options['maxiter'] = 20
                ##################################################################
                # Assemble at the group level. Default assembled jac type is 'csc'
                ##################################################################
                self.options['assembled_jac_type'] = 'csc'
                self.linear_solver = om.DirectSolver(assemble_jac=True)

        p = om.Problem()
        model = p.model

        model.add_subsystem('circuit', Circuit())

        p.setup()

        p.set_val('circuit.I_in', 0.1)
        p.set_val('circuit.Vg', 0.)

        # set some initial guesses
        p.set_val('circuit.n1.V', 10.)
        p.set_val('circuit.n2.V', 1.)

        p.run_model()

        assert_near_equal(p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_near_equal(p['circuit.n2.V'], 0.71278185, 1e-5)
        assert_near_equal(p['circuit.R1.I'], 0.09908047, 1e-5)
        assert_near_equal(p['circuit.R2.I'], 0.00091953, 1e-5)
        assert_near_equal(p['circuit.D1.I'], 0.00091953, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_near_equal(p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_plain_newton(self):

        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Resistor, Diode, Node

        class Circuit(om.Group):

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

                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.nonlinear_solver.options['iprint'] = 2
                self.nonlinear_solver.options['maxiter'] = 20
                self.linear_solver = om.DirectSolver()

        p = om.Problem()
        model = p.model

        model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', om.IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1.

        p.run_model()

        assert_near_equal(p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_near_equal(p['circuit.n2.V'], 0.71278185, 1e-5)
        assert_near_equal(p['circuit.R1.I'], 0.09908047, 1e-5)
        assert_near_equal(p['circuit.R2.I'], 0.00091953, 1e-5)
        assert_near_equal(p['circuit.D1.I'], 0.00091953, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_near_equal(p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_plain_newton_many_iter(self):

        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = om.Problem()
        model = p.model

        model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', om.IndepVarComp('I', 0.1, units='A'))
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

        assert_near_equal(p['circuit.n1.V'], 9.98744708, 1e-5)
        assert_near_equal(p['circuit.n2.V'], 8.73215484, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_near_equal(p['circuit.R1.I'] + p['circuit.D1.I'], 0.09987447, 1e-6)

    def test_circuit_advanced_newton(self):
        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = om.Problem()
        model = p.model

        model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', om.IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # you can change the NewtonSolver settings in circuit after setup is called
        newton = p.model.circuit.nonlinear_solver
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options['maxiter'] = 10
        newton.linesearch.options['iprint'] = 2

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        p.run_model()

        assert_near_equal(p['circuit.n1.V'], 9.90804735, 1e-5)
        assert_near_equal(p['circuit.n2.V'], 0.71278185, 1e-5)
        assert_near_equal(p['circuit.R1.I'], 0.09908047, 1e-5)
        assert_near_equal(p['circuit.R2.I'], 0.00091953, 1e-5)
        assert_near_equal(p['circuit.D1.I'], 0.00091953, 1e-5)

        # sanity check: should sum to .1 Amps
        assert_near_equal(p['circuit.R1.I'] + p['circuit.D1.I'], .1, 1e-6)

    def test_circuit_voltage_source(self):
        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = om.Problem()
        model = p.model

        model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))

        # replacing the fixed current source with a BalanceComp to represent a fixed Voltage source
        # model.add_subsystem('source', om.IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('batt', om.IndepVarComp('V', 1.5, units='V'))
        bal = model.add_subsystem('batt_balance', om.BalanceComp())
        bal.add_balance('I', units='A', eq_units='V')

        model.add_subsystem('circuit', Circuit())
        model.add_subsystem('batt_deltaV', om.ExecComp('dV = V1 - V2', V1={'units':'V'},
                                                       V2={'units':'V'}, dV={'units':'V'}))

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
        p.model.circuit.nonlinear_solver = om.NonlinearRunOnce()
        p.model.circuit.linear_solver = om.LinearRunOnce()

        # Put Newton at the top so it can also converge the new BalanceComp residual
        newton = p.model.nonlinear_solver = om.NewtonSolver()
        p.model.linear_solver = om.DirectSolver()
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 20
        newton.options['solve_subsystems'] = True
        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options['maxiter'] = 10
        newton.linesearch.options['iprint'] = 2

        # set initial guesses from the current source problem
        p['circuit.n1.V'] = 9.8
        p['circuit.n2.V'] = .7

        p.run_model()

        assert_near_equal(p['circuit.n1.V'], 1.5, 1e-5)
        assert_near_equal(p['circuit.n2.V'], 0.65113362, 1e-5)
        assert_near_equal(p['circuit.R1.I'], 0.015, 1e-5)
        assert_near_equal(p['circuit.R2.I'], 8.48866375e-05, 1e-5)
        assert_near_equal(p['circuit.D1.I'], 8.48866375e-05, 1e-5)


if __name__ == "__main__":
    unittest.main()
