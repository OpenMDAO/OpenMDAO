""" Test the ExternalCodeComp. """
import os
import sys
import shutil
import tempfile
import unittest

from scipy.optimize import fsolve

import openmdao.api as om
from openmdao.components.external_code_comp import STDOUT

from openmdao.utils.assert_utils import assert_near_equal, assert_warning

DIRECTORY = os.path.dirname((os.path.abspath(__file__)))


# These next three functions are used by test_simple_external_code_implicit_comp_with_solver
def area_ratio_explicit(Mach):
    """isentropic relationship between area ratio and Mach number"""
    gamma = 1.4
    gamma_p_1 = gamma + 1
    gamma_m_1 = gamma - 1
    exponent = gamma_p_1 / (2 * gamma_m_1)

    return (gamma_p_1 / 2.) ** -exponent * ((1 + gamma_m_1 / 2. * Mach ** 2) ** exponent) / Mach


def mach_residual(Mach, area_ratio_target):
    return area_ratio_target - area_ratio_explicit(Mach)


def mach_solve(area_ratio, super_sonic=False):
    if super_sonic:
        initial_guess = 4
    else:
        initial_guess = .1

    mach = fsolve(func=mach_residual, x0=initial_guess, args=(area_ratio,))[0]

    return mach


class TestExternalCodeComp(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        os.chdir(self.tempdir)
        shutil.copy(os.path.join(DIRECTORY, 'extcode_example.py'),
                    os.path.join(self.tempdir, 'extcode_example.py'))

        self.prob = om.Problem()

        self.extcode = self.prob.model.add_subsystem('extcode', om.ExternalCodeComp())

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_normal(self):
        self.extcode.options['command'] = [
            sys.executable, 'extcode_example.py', 'extcode.out'
        ]

        self.extcode.options['external_input_files'] = ['extcode_example.py']
        self.extcode.options['external_output_files'] = ['extcode.out']

        self.prob.setup(check=True)
        self.prob.run_model()

        with open('extcode.out', 'r') as f:
            self.assertEqual(f.read(), 'test data\n')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows-specific test.')
    def test_normal_bat(self):
        batch_script = '\n'.join([
            "@echo off",
            "rem usage: extcode.bat output_filename",
            "rem",
            "rem Just write 'test data' to the specified output file",
            "",
            "set DATA=test data",
            "set OUT_FILE=%1",
            "",
            "echo %DATA%>>%OUT_FILE%"
        ])
        with open('extcode.bat', 'w') as f:
            f.write(batch_script)

        self.extcode.options['command'] = [
            'extcode.bat', 'extcode.out'
        ]

        self.extcode.options['external_input_files'] = ['extcode.bat']
        self.extcode.options['external_output_files'] = ['extcode.out']

        self.prob.setup(check=True)
        self.prob.run_model()

        with open('extcode.out', 'r') as f:
            self.assertEqual(f.read(), 'test data\n')

    def test_timeout_raise(self):
        self.extcode.options['command'] = [
            sys.executable, 'extcode_example.py', 'extcode.out', '--delay', '3'
        ]
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['extcode_example.py']

        self.prob.setup(check=True)
        try:
            self.prob.run_model()
        except om.AnalysisError as exc:
            self.assertEqual(str(exc), 'Timed out after 1.0 sec.')
        else:
            self.fail('Expected AnalysisError')

    def test_error_code_raise(self):
        self.extcode.options['command'] = [
            sys.executable, 'extcode_example.py', 'extcode.out', '--delay', '-3'
        ]
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['extcode_example.py']

        self.prob.setup(check=True)
        try:
            self.prob.run_model()
        except RuntimeError as exc:
            self.assertTrue('Traceback' in str(exc),
                            "no traceback found in '%s'" % str(exc))
            self.assertEqual(self.extcode.return_code, 1)
        else:
            self.fail('Expected RuntimeError')

    def test_error_code_soft(self):
        self.extcode.options['command'] = [
            sys.executable, 'extcode_example.py', 'extcode.out', '--delay', '-3'
        ]
        self.extcode.options['timeout'] = 1.0
        self.extcode.options['fail_hard'] = False

        self.extcode.options['external_input_files'] = ['extcode_example.py']

        self.prob.setup(check=True)
        try:
            self.prob.run_model()
        except om.AnalysisError as err:
            self.assertTrue("delay must be >= 0" in str(err),
                            "expected 'delay must be >= 0' to be in '%s'" % str(err))
            self.assertTrue('Traceback' in str(err),
                            "no traceback found in '%s'" % str(err))
        else:
            self.fail("AnalysisError expected")

    def test_allowed_return_code(self):
        self.extcode.options['allowed_return_codes'] = set(range(5))
        self.extcode.options['command'] = [
            sys.executable, 'extcode_example.py', 'extcode.out', '--return_code', '4'
        ]

        self.extcode.options['external_input_files'] = ['extcode_example.py']

        self.prob.setup(check=True)
        self.prob.run_model()

    def test_disallowed_return_code(self):
        self.extcode.options['allowed_return_codes'] = list(range(5))
        self.extcode.options['command'] = [
            sys.executable, 'extcode_example.py', 'extcode.out', '--return_code', '7'
        ]

        self.extcode.options['external_input_files'] = ['extcode_example.py']

        self.prob.setup(check=True)
        try:
            self.prob.run_model()
        except RuntimeError as err:
            self.assertTrue("return_code = 7" in str(err),
                            "expected 'return_code = 7' to be in '%s'" % str(err))
        else:
            self.fail("RuntimeError expected")

    def test_badcmd(self):
        # Set command to nonexistant path.
        self.extcode.options['command'] = ['no-such-command']

        self.prob.setup()
        try:
            self.prob.run_model()
        except ValueError as exc:
            msg = "The command to be executed, 'no-such-command', cannot be found"
            self.assertEqual(str(exc), msg)
            self.assertEqual(self.extcode.return_code, -999999)
        else:
            self.fail('Expected ValueError')

    def test_nullcmd(self):
        self.extcode.stdout = 'nullcmd.out'
        self.extcode.stderr = STDOUT

        self.prob.setup()
        try:
            self.prob.run_model()
        except ValueError as exc:
            self.assertEqual(str(exc), 'Empty command list')
        else:
            self.fail('Expected ValueError')
        finally:
            if os.path.exists(self.extcode.stdout):
                os.remove(self.extcode.stdout)

    def test_env_vars(self):
        self.extcode.options['env_vars'] = {'TEST_ENV_VAR': 'SOME_ENV_VAR_VALUE'}
        self.extcode.options['command'] = [
            sys.executable, 'extcode_example.py', 'extcode.out', '--write_test_env_var'
        ]

        self.prob.setup(check=True)
        self.prob.run_model()

        # Check to see if output file contains the env var value
        with open(os.path.join(self.tempdir, 'extcode.out'), 'r') as out:
            file_contents = out.read()
        self.assertTrue('SOME_ENV_VAR_VALUE' in file_contents,
                        "'SOME_ENV_VAR_VALUE' missing from '%s'" % file_contents)


class TestExternalCodeCompArgs(unittest.TestCase):

    def test_kwargs(self):
        # check kwargs are passed to options
        extcode = om.ExternalCodeComp(poll_delay=999)

        self.assertTrue(extcode.options['poll_delay'] == 999)

        # check subclass kwargs are also passed to options
        class MyComp(om.ExternalCodeComp):
            def initialize(self):
                self.options.declare('my_arg', 'foo', desc='subclass option')

        my_comp = MyComp(poll_delay=999, my_arg='bar')

        self.assertTrue(my_comp.options['poll_delay'] == 999)
        self.assertTrue(my_comp.options['my_arg'] == 'bar')

        # check that options are those declared in both classes
        extcode_opts = set(extcode.options._dict.keys())
        my_comp_opts = set(my_comp.options._dict.keys())

        self.assertEqual(my_comp_opts.difference(extcode_opts), set(('my_arg',)))


class ParaboloidExternalCodeComp(om.ExternalCodeComp):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file]
        self.options['external_output_files'] = [self.output_file]

        # If you want to write your command as a list, the code below will also work.
        # self.options['command'] = [
        #     sys.executable, 'extcode_paraboloid.py', self.input_file, self.output_file
        # ]

        self.options['command'] = ('python extcode_paraboloid.py {} {}').format(self.input_file, self.output_file)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x, y))

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCodeComp, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy


class ParaboloidExternalCodeCompFD(om.ExternalCodeComp):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file]
        self.options['external_output_files'] = [self.output_file]

        self.options['command'] = [
            sys.executable, 'extcode_paraboloid.py', self.input_file, self.output_file
        ]

        # this external code does not provide derivatives, use finite difference
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x, y))

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCodeCompFD, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy


class ParaboloidExternalCodeCompDerivs(om.ExternalCodeComp):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'
        self.derivs_file = 'paraboloid_derivs.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file]
        self.options['external_output_files'] = [self.output_file, self.derivs_file]

        self.options['command'] = [
            sys.executable, 'extcode_paraboloid_derivs.py',
            self.input_file, self.output_file, self.derivs_file
        ]

        # this external code does provide derivatives
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x, y))

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCodeCompDerivs, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy

    def compute_partials(self, inputs, partials):
        outputs = {}

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCodeCompDerivs, self).compute(inputs, outputs)

        # parse the derivs file from the external code and set partials
        with open(self.derivs_file, 'r') as derivs_file:
            partials['f_xy', 'x'] = float(derivs_file.readline())
            partials['f_xy', 'y'] = float(derivs_file.readline())


class TestExternalCodeCompFeature(unittest.TestCase):

    def setUp(self):
        import os
        import shutil
        import tempfile

        # get the directory where the needed support files are located
        import openmdao.components.tests.test_external_code_comp as extcode_test
        DIRECTORY = os.path.dirname((os.path.abspath(extcode_test.__file__)))

        # change to temp dir
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        os.chdir(self.tempdir)

        # copy required files to temp dir
        files = ['extcode_paraboloid.py', 'extcode_paraboloid_derivs.py']
        for filename in files:
            shutil.copy(os.path.join(DIRECTORY, filename),
                        os.path.join(self.tempdir, filename))

    def tearDown(self):
        # destroy the evidence
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_main(self):
        import openmdao.api as om
        from openmdao.components.tests.test_external_code_comp import ParaboloidExternalCodeComp

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', ParaboloidExternalCodeComp(), promotes_inputs=['x', 'y'])

        # run the ExternalCodeComp Component
        prob.setup()

        # Set input values
        prob.set_val('p.x', 3.0)
        prob.set_val('p.y', -4.0)

        prob.run_model()

        # print the output
        self.assertEqual(prob.get_val('p.f_xy'), -15.0)

    def test_optimize_fd(self):
        import openmdao.api as om
        from openmdao.components.tests.test_external_code_comp import ParaboloidExternalCodeCompFD

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', ParaboloidExternalCodeCompFD())

        # find optimal solution with SciPy optimize
        # solution (minimum): x = 6.6667; y = -7.3333
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('p.x', lower=-50, upper=50)
        prob.model.add_design_var('p.y', lower=-50, upper=50)

        prob.model.add_objective('p.f_xy')

        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()

        # Set input values
        prob.set_val('p.x', 3.0)
        prob.set_val('p.y', -4.0)

        prob.run_driver()

        assert_near_equal(prob.get_val('p.x'), 6.66666667, 1e-6)
        assert_near_equal(prob.get_val('p.y'), -7.3333333, 1e-6)

    def test_optimize_derivs(self):
        import openmdao.api as om
        from openmdao.components.tests.test_external_code_comp import ParaboloidExternalCodeCompDerivs

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', ParaboloidExternalCodeCompDerivs())

        # find optimal solution with SciPy optimize
        # solution (minimum): x = 6.6667; y = -7.3333
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('p.x', lower=-50, upper=50)
        prob.model.add_design_var('p.y', lower=-50, upper=50)

        prob.model.add_objective('p.f_xy')

        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()

        # Set input values
        prob.set_val('p.x', 3.0)
        prob.set_val('p.y', -4.0)

        prob.run_driver()

        assert_near_equal(prob.get_val('p.x'), 6.66666667, 1e-6)
        assert_near_equal(prob.get_val('p.y'), -7.3333333, 1e-6)


class TestExternalCodeImplicitCompFeature(unittest.TestCase):

    def setUp(self):
        import os
        import shutil
        import tempfile

        # get the directory where the needed support files are located
        import openmdao.components.tests.test_external_code_comp as extcode_test
        DIRECTORY = os.path.dirname((os.path.abspath(extcode_test.__file__)))

        # change to temp dir
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        os.chdir(self.tempdir)

        # copy required files to temp dir
        files = ['extcode_resistor.py', 'extcode_node.py', 'extcode_mach.py']
        for filename in files:
            shutil.copy(os.path.join(DIRECTORY, filename),
                        os.path.join(self.tempdir, filename))

    def tearDown(self):
        # destroy the evidence
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_simple_external_code_implicit_comp(self):
        import sys
        import openmdao.api as om

        class MachExternalCodeComp(om.ExternalCodeImplicitComp):

            def initialize(self):
                self.options.declare('super_sonic', types=bool)

            def setup(self):
                self.add_input('area_ratio', val=1.0, units=None)
                self.add_output('mach', val=1., units=None)
                self.declare_partials(of='mach', wrt='area_ratio', method='fd')

                self.input_file = 'mach_input.dat'
                self.output_file = 'mach_output.dat'

                # providing these are optional; the component will verify that any input
                # files exist before execution and that the output files exist after.
                self.options['external_input_files'] = [self.input_file]
                self.options['external_output_files'] = [self.output_file]


                self.options['command_apply'] = [
                    sys.executable, 'extcode_mach.py', self.input_file, self.output_file,
                ]
                self.options['command_solve'] = [
                    sys.executable, 'extcode_mach.py', self.input_file, self.output_file,
                ]

                # If you want to write your own string command, the code below will also work.
                # self.options['command_apply'] = ('python extcode_mach.py {} {}').format(self.input_file, self.output_file)

            def apply_nonlinear(self, inputs, outputs, residuals):
                with open(self.input_file, 'w') as input_file:
                    input_file.write('residuals\n')
                    input_file.write('{}\n'.format(inputs['area_ratio'][0]))
                    input_file.write('{}\n'.format(outputs['mach'][0]))

                # the parent apply_nonlinear function actually runs the external code
                super(MachExternalCodeComp, self).apply_nonlinear(inputs, outputs, residuals)

                # parse the output file from the external code and set the value of mach
                with open(self.output_file, 'r') as output_file:
                    mach = float(output_file.read())
                residuals['mach'] = mach

            def solve_nonlinear(self, inputs, outputs):
                with open(self.input_file, 'w') as input_file:
                    input_file.write('outputs\n')
                    input_file.write('{}\n'.format(inputs['area_ratio'][0]))
                    input_file.write('{}\n'.format(self.options['super_sonic']))
                # the parent apply_nonlinear function actually runs the external code
                super(MachExternalCodeComp, self).solve_nonlinear(inputs, outputs)

                # parse the output file from the external code and set the value of mach
                with open(self.output_file, 'r') as output_file:
                    mach = float(output_file.read())
                outputs['mach'] = mach

        group = om.Group()
        mach_comp = group.add_subsystem('comp', MachExternalCodeComp(), promotes=['*'])
        prob = om.Problem(model=group)
        group.nonlinear_solver = om.NewtonSolver()
        group.nonlinear_solver.options['solve_subsystems'] = True
        group.nonlinear_solver.options['iprint'] = 0
        group.nonlinear_solver.options['maxiter'] = 20
        group.linear_solver = om.DirectSolver()

        prob.setup()

        area_ratio = 1.3
        super_sonic = False
        prob.set_val('area_ratio', area_ratio)
        mach_comp.options['super_sonic'] = super_sonic
        prob.run_model()
        assert_near_equal(prob.get_val('mach'), mach_solve(area_ratio, super_sonic=super_sonic), 1e-8)

        area_ratio = 1.3
        super_sonic = True
        prob.set_val('area_ratio', area_ratio)
        mach_comp.options['super_sonic'] = super_sonic
        prob.run_model()
        assert_near_equal(prob.get_val('mach'), mach_solve(area_ratio, super_sonic=super_sonic), 1e-8)

if __name__ == "__main__":
    unittest.main()
