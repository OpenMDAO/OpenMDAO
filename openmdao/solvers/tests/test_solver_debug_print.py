"""Tests the `debug_print` option for Nonlinear solvers."""

import os
import re
import sys
import shutil
import tempfile

import unittest
from distutils.version import LooseVersion
from io import StringIO

import numpy as np

import openmdao.api as om
from openmdao.test_suite.scripts.circuit_analysis import Circuit

from openmdao.utils.general_utils import run_model
from openmdao.utils.general_utils import printoptions

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

nonlinear_solvers = [
    om.NonlinearBlockGS,
    om.NonlinearBlockJac,
    om.NewtonSolver,
    om.BroydenSolver
]


class TestNonlinearSolvers(unittest.TestCase):
    def setUp(self):
        import re
        import os
        from tempfile import mkdtemp

        # perform test in temporary directory
        self.startdir = os.getcwd()
        self.tempdir = mkdtemp(prefix='test_solver')
        os.chdir(self.tempdir)

        # iteration coordinate, file name and variable data are common for all tests
        coord = 'rank0:root._solve_nonlinear|0|NLRunOnce|0|circuit._solve_nonlinear|0'
        self.filename = 'solver_errors.0.out'

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
        import os
        from shutil import rmtree

        # clean up the temporary directory
        os.chdir(self.startdir)
        try:
            rmtree(self.tempdir)
        except OSError:
            pass

    @parameterized.expand([
        [solver.__name__, solver] for solver in nonlinear_solvers
    ])
    def test_solver_debug_print(self, name, solver):
        p = om.Problem()
        model = p.model

        model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', om.IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()
        nl = model.circuit.nonlinear_solver = solver()

        nl.options['debug_print'] = True
        nl.options['err_on_non_converge'] = True

        if name == 'NonlinearBlockGS':
            nl.options['use_apply_nonlinear'] = True

        if name == 'NewtonSolver':
            nl.options['solve_subsystems'] = True

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
            output = run_model(p, ignore_exception=True)

        expected_output = '\n'.join([
            self.expected_data,
            "Inputs and outputs at start of iteration "
            "have been saved to '%s'.\n" % self.filename
        ])

        self.assertEqual(output, expected_output)

        with open(self.filename, 'r') as f:
            self.assertEqual(f.read(), self.expected_data)

        # setup & run again to make sure there is no error due to existing file
        p.setup()
        with printoptions(**opts):
            run_model(p, ignore_exception=False)

    def test_solver_debug_print_feature(self):
        from distutils.version import LooseVersion
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Circuit
        from openmdao.utils.general_utils import printoptions

        p = om.Problem()
        model = p.model

        model.add_subsystem('circuit', Circuit())

        p.setup()

        nl = model.circuit.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        nl.options['iprint'] = 2
        nl.options['debug_print'] = True
        nl.options['err_on_non_converge'] = True

        # set some poor initial guesses so that we don't converge
        p.set_val('circuit.I_in', 0.1, units='A')
        p.set_val('circuit.Vg', 0.0, units='V')
        p.set_val('circuit.n1.V', 10.)
        p.set_val('circuit.n2.V', 1e-3)

        opts = {}
        # formatting has changed in numpy 1.14 and beyond.
        if LooseVersion(np.__version__) >= LooseVersion("1.14"):
            opts["legacy"] = '1.13'

        with printoptions(**opts):
            # run the model
            try:
                p.run_model()
            except om.AnalysisError:
                pass

        with open(self.filename, 'r') as f:
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
        prob = om.Problem()
        model = prob.model

        comp = om.IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=om.Group())
        teg.add_subsystem('dynamics', om.ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = om.BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = om.DirectSolver()

        teg.nonlinear_solver = om.NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4
        teg.nonlinear_solver.options['debug_print'] = True

        prob.setup()
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
        expected_msg = "Singular entry found in 'thrust_equilibrium_group' <class Group> for row associated with state/residual 'thrust' ('thrust_equilibrium_group.thrust_bal.thrust') index 0."
        self.assertEqual(expected_msg, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
