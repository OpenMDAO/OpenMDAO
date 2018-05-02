""" Test the ExternalCodeComp. """
from __future__ import print_function

import os
import shutil
import tempfile
import unittest

from openmdao.api import Problem, Group, ExternalCodeComp, AnalysisError
from openmdao.components.external_code_comp import STDOUT

from openmdao.utils.assert_utils import assert_rel_error

DIRECTORY = os.path.dirname((os.path.abspath(__file__)))


class TestExternalCodeComp(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        os.chdir(self.tempdir)
        shutil.copy(os.path.join(DIRECTORY, 'extcode_example.py'),
                    os.path.join(self.tempdir, 'extcode_example.py'))

        self.prob = Problem()

        self.extcode = self.prob.model.add_subsystem('extcode', ExternalCodeComp())

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_normal(self):
        self.extcode.options['command'] = [
            'python', 'extcode_example.py', 'extcode.out'
        ]

        self.extcode.options['external_input_files'] = ['extcode_example.py',]
        self.extcode.options['external_output_files'] = ['extcode.out',]

        dev_null = open(os.devnull, 'w')
        self.prob.setup(check=True)
        self.prob.run_model()

    def test_timeout_raise(self):
        self.extcode.options['command'] = [
            'python', 'extcode_example.py', 'extcode.out', '--delay', '3'
        ]
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['extcode_example.py', ]

        dev_null = open(os.devnull, 'w')
        self.prob.setup(check=True)
        try:
            self.prob.run_model()
        except AnalysisError as exc:
            self.assertEqual(str(exc), 'Timed out after 1.0 sec.')
        else:
            self.fail('Expected AnalysisError')

    def test_error_code_raise(self):
        self.extcode.options['command'] = [
            'python', 'extcode_example.py', 'extcode.out', '--delay', '-3'
        ]
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['extcode_example.py', ]

        dev_null = open(os.devnull, 'w')
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
            'python', 'extcode_example.py', 'extcode.out', '--delay', '-3'
        ]
        self.extcode.options['timeout'] = 1.0
        self.extcode.options['fail_hard'] = False

        self.extcode.options['external_input_files'] = ['extcode_example.py', ]

        dev_null = open(os.devnull, 'w')
        self.prob.setup(check=True)
        try:
            self.prob.run_model()
        except AnalysisError as err:
            self.assertTrue("delay must be >= 0" in str(err),
                            "expected 'delay must be >= 0' to be in '%s'" % str(err))
            self.assertTrue('Traceback' in str(err),
                            "no traceback found in '%s'" % str(err))
        else:
            self.fail("AnalysisError expected")

    def test_allowed_return_code(self):
        self.extcode.options['allowed_return_codes'] = set(range(5))
        self.extcode.options['command'] = [
            'python', 'extcode_example.py', 'extcode.out', '--return_code', '4'
        ]

        self.extcode.options['external_input_files'] = ['extcode_example.py', ]

        dev_null = open(os.devnull, 'w')
        self.prob.setup(check=True)
        self.prob.run_model()

    def test_disallowed_return_code(self):
        self.extcode.options['allowed_return_codes'] = list(range(5))
        self.extcode.options['command'] = [
            'python', 'extcode_example.py', 'extcode.out', '--return_code', '7'
        ]

        self.extcode.options['external_input_files'] = ['extcode_example.py', ]

        dev_null = open(os.devnull, 'w')
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
        self.extcode.options['command'] = ['no-such-command', ]

        self.prob.setup(check=False)
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

        self.prob.setup(check=False)
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
            'python', 'extcode_example.py', 'extcode.out', '--write_test_env_var'
        ]

        dev_null = open(os.devnull, 'w')
        self.prob.setup(check=True)
        self.prob.run_model()

        # Check to see if output file contains the env var value
        with open(os.path.join(self.tempdir, 'extcode.out'), 'r') as out:
            file_contents = out.read()
        self.assertTrue('SOME_ENV_VAR_VALUE' in file_contents,
                        "'SOME_ENV_VAR_VALUE' missing from '%s'" % file_contents)


class TestExternalCodeCompArgs(unittest.TestCase):

    def test_kwargs(self):
        extcode = ExternalCodeComp(poll_delay=999)

        self.assertTrue(extcode.options['poll_delay'] == 999)


class ParaboloidExternalCodeComp(ExternalCodeComp):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]

        self.options['command'] = [
            'python', 'extcode_paraboloid.py', self.input_file, self.output_file
        ]

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x,y))

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCodeComp, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy


class ParaboloidExternalCodeCompFD(ExternalCodeComp):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]

        self.options['command'] = [
            'python', 'extcode_paraboloid.py', self.input_file, self.output_file
        ]

        # this external code does not provide derivatives, use finite difference
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x,y))

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCodeCompFD, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy


class ParaboloidExternalCodeCompDerivs(ExternalCodeComp):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'
        self.derivs_file = 'paraboloid_derivs.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file, self.derivs_file]

        self.options['command'] = [
            'python', 'extcode_paraboloid_derivs.py',
            self.input_file, self.output_file, self.derivs_file
        ]

        # this external code does provide derivatives
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x,y))

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
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.components.tests.test_external_code_comp import ParaboloidExternalCodeComp

        prob = Problem()
        model = prob.model

        # create and connect inputs
        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('p', ParaboloidExternalCodeComp())

        model.connect('p1.x', 'p.x')
        model.connect('p2.y', 'p.y')

        # run the ExternalCodeComp Component
        prob.setup()
        prob.run_model()

        # print the output
        self.assertEqual(prob['p.f_xy'], -15.0)

    def test_optimize_fd(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.api import ScipyOptimizeDriver
        from openmdao.components.tests.test_external_code_comp import ParaboloidExternalCodeCompFD

        prob = Problem()
        model = prob.model

        # create and connect inputs
        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('p', ParaboloidExternalCodeCompFD())

        model.connect('p1.x', 'p.x')
        model.connect('p2.y', 'p.y')

        # find optimal solution with SciPy optimize
        # solution (minimum): x = 6.6667; y = -7.3333
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('p1.x', lower=-50, upper=50)
        prob.model.add_design_var('p2.y', lower=-50, upper=50)

        prob.model.add_objective('p.f_xy')

        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, prob['p1.x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['p2.y'], -7.3333333, 1e-6)

    def test_optimize_derivs(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.api import ScipyOptimizeDriver
        from openmdao.components.tests.test_external_code_comp import ParaboloidExternalCodeCompDerivs

        prob = Problem()
        model = prob.model

        # create and connect inputs
        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('p', ParaboloidExternalCodeCompDerivs())

        model.connect('p1.x', 'p.x')
        model.connect('p2.y', 'p.y')

        # find optimal solution with SciPy optimize
        # solution (minimum): x = 6.6667; y = -7.3333
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('p1.x', lower=-50, upper=50)
        prob.model.add_design_var('p2.y', lower=-50, upper=50)

        prob.model.add_objective('p.f_xy')

        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, prob['p1.x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['p2.y'], -7.3333333, 1e-6)

# ------------------------------------------------------
# run same test as above, only with the deprecated component,
# to ensure we get the warning and the correct answer.
# self-contained, to be removed when class name goes away.
from openmdao.api import ExternalCode
import warnings


class DeprecatedExternalCodeForTesting(ExternalCode):
    def __init__(self):
        super(DeprecatedExternalCodeForTesting, self).__init__()


class TestDeprecatedExternalCode(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        os.chdir(self.tempdir)
        shutil.copy(os.path.join(DIRECTORY, 'extcode_example.py'),
                    os.path.join(self.tempdir, 'extcode_example.py'))

        with warnings.catch_warnings(record=True) as w:
            self.extcode = DeprecatedExternalCodeForTesting()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), "'ExternalCode' has been deprecated. Use 'ExternalCodeComp' instead.")

        self.prob = Problem()

        self.prob.model.add_subsystem('extcode', self.extcode)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_normal(self):
        self.extcode.options['command'] = [
            'python', 'extcode_example.py', 'extcode.out'
        ]

        self.extcode.options['external_input_files'] = ['extcode_example.py',]
        self.extcode.options['external_output_files'] = ['extcode.out',]

        dev_null = open(os.devnull, 'w')
        self.prob.setup(check=True)
        self.prob.run_model()


if __name__ == "__main__":
    unittest.main()
