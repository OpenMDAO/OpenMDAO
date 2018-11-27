"""Tests the `debug_print` option for Nonlinear solvers."""

from __future__ import division, print_function

from six import StringIO

import os
import re
import sys
import shutil
import tempfile

import unittest
from distutils.version import LooseVersion

import numpy as np

from openmdao.api import Problem, IndepVarComp, ExecComp, Group, BalanceComp

from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.nonlinear.broyden import BroydenSolver
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from openmdao.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac

from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import run_model
from openmdao.utils.general_utils import printoptions

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

nonlinear_solvers = [
    NonlinearBlockGS,
    NonlinearBlockJac,
    NewtonSolver,
    BroydenSolver
]


class TestNonlinearSolvers(unittest.TestCase):
    def setUp(self):
        # perform test in temporary directory
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_solver')
        os.chdir(self.tempdir)

        # iteration coordinate, file name and variable data are common for all tests
        coord = 'rank0:root._solve_nonlinear|0|NLRunOnce|0|circuit._solve_nonlinear|0'
        filename = coord.replace('._solve_nonlinear', '')
        self.filename = re.sub('[^0-9a-zA-Z]', '_', filename) + '.dat'

        self.expected_data = '\n'.join([
            "",
            "# Inputs and outputs at start of iteration '%s':" % coord,
            "",
            "# nonlinear inputs",
            "{'circuit.D1.V_in': array([ 1.]),",
            " 'circuit.D1.V_out': array([ 0.]),",
            " 'circuit.R1.V_in': array([ 1.]),",
            " 'circuit.R1.V_out': array([ 0.]),",
            " 'circuit.R2.V_in': array([ 1.]),",
            " 'circuit.R2.V_out': array([ 1.]),",
            " 'circuit.n1.I_in:0': array([ 0.1]),",
            " 'circuit.n1.I_out:0': array([ 1.]),",
            " 'circuit.n1.I_out:1': array([ 1.]),",
            " 'circuit.n2.I_in:0': array([ 1.]),",
            " 'circuit.n2.I_out:0': array([ 1.])}",
            "",
            "# nonlinear outputs",
            "{'circuit.D1.I': array([ 1.]),",
            " 'circuit.R1.I': array([ 1.]),",
            " 'circuit.R2.I': array([ 1.]),",
            " 'circuit.n1.V': array([ 10.]),",
            " 'circuit.n2.V': array([ 0.001])}",
            ""
        ])

    def tearDown(self):
        # clean up the temporary directory
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @parameterized.expand([
        [solver.__name__, solver] for solver in nonlinear_solvers
    ])
    def test_solver_debug_print(self, name, solver):
        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        nl = model.circuit.nonlinear_solver = solver()

        nl.options['debug_print'] = True

        # suppress solver output for test
        nl.options['iprint'] = model.circuit.linear_solver.options['iprint'] = -1

        # For Broydensolver, don't calc Jacobian
        try:
            nl.options['compute_jacobian'] = False
        except KeyError:
            pass

        # set some poor initial guesses so that we don't converge
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        opts = {}
        # formatting has changed in numpy 1.14 and beyond.
        if LooseVersion(np.__version__) >= LooseVersion("1.14"):
            opts["legacy"] = '1.13'

        with printoptions(**opts):
            # run the model and check for expected output file
            output = run_model(p)

        expected_output = '\n'.join([
            self.expected_data,
            "Inputs and outputs at start of iteration "
            "have been saved to '%s'.\n" % self.filename
        ])

        self.assertEqual(output, expected_output)

        with open(self.filename, 'r') as f:
            self.assertEqual(f.read(), self.expected_data)

    def test_solver_debug_print_feature(self):
        from openmdao.api import Problem, IndepVarComp, NewtonSolver
        from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit
        from openmdao.utils.general_utils import printoptions

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        nl = model.circuit.nonlinear_solver = NewtonSolver()

        nl.options['iprint'] = 2
        nl.options['debug_print'] = True

        # set some poor initial guesses so that we don't converge
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        opts = {}
        # formatting has changed in numpy 1.14 and beyond.
        if LooseVersion(np.__version__) >= LooseVersion("1.14"):
            opts["legacy"] = '1.13'

        with printoptions(**opts):
            # run the model
            p.run_model()

        with open('rank0_root_0_NLRunOnce_0_circuit_0.dat', 'r') as f:
            self.assertEqual(f.read(), self.expected_data)


class TestNonlinearSolversIsolated(unittest.TestCase):
    """
    This test needs to run isolated to preclude interactions in the underlying
    `warnings` module that is used to raise the singular entry error.
    """
    ISOLATED = True

    def setUp(self):
        # perform test in temporary directory
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_solver')
        os.chdir(self.tempdir)

    def tearDown(self):
        # clean up the temporary directory
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_debug_after_raised_error(self):
        prob = Problem()
        model = prob.model

        comp = IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=Group())
        teg.add_subsystem('dynamics', ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = DirectSolver()

        teg.nonlinear_solver = NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4
        teg.nonlinear_solver.options['debug_print'] = True

        prob.setup(check=False)
        prob.set_solver_print(level=0)

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        sys.stdout = stdout

        output = strout.getvalue()
        target = "'thrust_equilibrium_group.thrust_bal.thrust'"
        self.assertTrue(target in output, msg=target + "NOT FOUND IN" + output)

        # Make sure exception is unchanged.
        expected_msg = "Singular entry found in 'thrust_equilibrium_group' for column associated with state/residual 'thrust'."
        self.assertEqual(expected_msg, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
