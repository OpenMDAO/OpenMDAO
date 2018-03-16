"""Runs parametric tests over multiple solvers."""

from __future__ import division, print_function

import unittest
import re

from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp

from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from openmdao.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from openmdao.solvers.nonlinear.newton import NewtonSolver

from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

from openmdao.utils.assert_utils import assert_rel_error

from nose_parameterized import parameterized


nonlinear_solvers = [
    NonlinearBlockGS,
    NonlinearBlockJac,
    NewtonSolver
]


class TestNonlinearSolvers(unittest.TestCase):
    @parameterized.expand([
        [solver.__name__, solver] for solver in nonlinear_solvers
    ])
    def test_solver(self, name, solver):
        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        nl = model.circuit.nonlinear_solver = solver()

        filename = name + '.dat'
        nl.options['err_output_file'] = filename

        # suppress solver output for test
        nl.options['iprint'] = model.circuit.linear_solver.options['iprint'] = -1

        # set some poor initial guesses so that we don't converge
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        p.run_model()

        expected_data = [
            '# inputs and outputs at start of %s iteration' % solver.SOLVER,
            '',
            '# nonlinear input vector',
            'circuit.D1.V_in = array([ 1.])',
            'circuit.D1.V_out = array([ 0.])',
            'circuit.R1.V_in = array([ 1.])',
            'circuit.R1.V_out = array([ 0.])',
            'circuit.R2.V_in = array([ 1.])',
            'circuit.R2.V_out = array([ 1.])',
            'circuit.n1.I_in:0 = array([ 0.1])',
            'circuit.n1.I_out:0 = array([ 1.])',
            'circuit.n1.I_out:1 = array([ 1.])',
            'circuit.n2.I_in:0 = array([ 1.])',
            'circuit.n2.I_out:0 = array([ 1.])',
            '',
            '# nonlinear output vector',
            'circuit.D1.I = array([ 1.])',
            'circuit.R1.I = array([ 1.])',
            'circuit.R2.I = array([ 1.])',
            'circuit.n1.V = array([ 10.])',
            'circuit.n2.V = array([ 0.001])'
        ]

        # make sure error file is generated as expected
        i = 0
        with open(filename, 'r') as f:
            for line in f:
                self.assertEqual(line.strip(), expected_data[i])
                i += 1


if __name__ == "__main__":
    unittest.main()
