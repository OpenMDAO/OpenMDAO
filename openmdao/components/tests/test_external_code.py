""" Test the ExternalCode. """
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import unittest

from openmdao.api import Problem, Group, ExternalCode, AnalysisError
from openmdao.components.external_code import STDOUT

DIRECTORY = os.path.dirname((os.path.abspath(__file__)))


class ExternalCodeForTesting(ExternalCode):
    def __init__(self):
        super(ExternalCodeForTesting, self).__init__()


class TestExternalCode(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        os.chdir(self.tempdir)
        shutil.copy(os.path.join(DIRECTORY, 'external_code_sample.py'),
                    os.path.join(self.tempdir, 'external_code_sample.py'))

        self.extcode = ExternalCodeForTesting()
        self.top = Problem()
        self.top.model = Group()

        self.top.model.add_subsystem('extcode', self.extcode)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_normal(self):
        self.extcode.options['command'] = ['python', 'external_code_sample.py', 'external_code_output.txt']

        self.extcode.options['external_input_files'] = ['external_code_sample.py',]
        self.extcode.options['external_output_files'] = ['external_code_output.txt',]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True)
        self.top.run_model()

    def test_timeout_raise(self):

        self.extcode.options['command'] = ['python', 'external_code_sample.py',
             'external_code_output.txt', '--delay', '3']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['external_code_sample.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True)
        try:
            self.top.run_model()
        except AnalysisError as exc:
            self.assertEqual(str(exc), 'Timed out after 1.0 sec.')
        else:
            self.fail('Expected AnalysisError')

    def test_error_code_raise(self):

        self.extcode.options['command'] = ['python', 'external_code_sample.py',
             'external_code_output.txt', '--delay', '-3']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['external_code_sample.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True)
        try:
            self.top.run_model()
        except RuntimeError as exc:
            self.assertTrue('Traceback' in str(exc),
                            "no traceback found in '%s'" % str(exc))
            self.assertEqual(self.extcode.return_code, 1)
        else:
            self.fail('Expected RuntimeError')

    def test_error_code_soft(self):

        self.extcode.options['command'] = ['python', 'external_code_sample.py',
             'external_code_output.txt', '--delay', '-3']
        self.extcode.options['timeout'] = 1.0
        self.extcode.options['fail_hard'] = False

        self.extcode.options['external_input_files'] = ['external_code_sample.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True)
        try:
            self.top.run_model()
        except AnalysisError as err:
            self.assertTrue("delay must be >= 0" in str(err),
                            "expected 'delay must be >= 0' to be in '%s'" % str(err))
            self.assertTrue('Traceback' in str(err),
                            "no traceback found in '%s'" % str(err))
        else:
            self.fail("AnalysisError expected")

    def test_badcmd(self):

        # Set command to nonexistant path.
        self.extcode.options['command'] = ['no-such-command', ]

        self.top.setup(check=False)
        try:
            self.top.run_model()
        except ValueError as exc:
            msg = "The command to be executed, 'no-such-command', cannot be found"
            self.assertEqual(str(exc), msg)
            self.assertEqual(self.extcode.return_code, -999999)
        else:
            self.fail('Expected ValueError')

    def test_nullcmd(self):

        self.extcode.stdout = 'nullcmd.out'
        self.extcode.stderr = STDOUT

        self.top.setup(check=False)
        try:
            self.top.run_model()
        except ValueError as exc:
            self.assertEqual(str(exc), 'Empty command list')
        else:
            self.fail('Expected ValueError')
        finally:
            if os.path.exists(self.extcode.stdout):
                os.remove(self.extcode.stdout)

    def test_env_vars(self):

        self.extcode.options['env_vars'] = {'TEST_ENV_VAR': 'SOME_ENV_VAR_VALUE'}
        self.extcode.options['command'] = ['python', 'external_code_sample.py', 'external_code_output.txt', '--write_test_env_var']

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True)
        self.top.run_model()

        # Check to see if output file contains the env var value
        with open(os.path.join(self.tempdir, 'external_code_output.txt'), 'r') as out:
            file_contents = out.read()
        self.assertTrue('SOME_ENV_VAR_VALUE' in file_contents,
                        "'SOME_ENV_VAR_VALUE' missing from '%s'" % file_contents)


class ParaboloidExternalCode(ExternalCode):
    def setup(self):

        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_filepath = 'paraboloid_input.dat'
        self.output_filepath = 'paraboloid_output.dat'

        #providing these is optional, but has the component check to make sure they are there
        self.options['external_input_files'] = [self.input_filepath,]
        self.options['external_output_files'] = [self.output_filepath,]

        self.options['command'] = ['python', 'external_code_feature_sample.py',
                                   self.input_filepath, self.output_filepath]

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # Generate the input file for the paraboloid external code
        with open(self.input_filepath, 'w') as input_file:
            input_file.write('%f\n%f\n' % (x,y))

        #parent solve_nonlinear function actually runs the external code
        super(ParaboloidExternalCode, self).compute(inputs, outputs)

        # Parse the output file from the external code and set the value of f_xy
        with open(self.output_filepath, 'r') as output_file:
            f_xy = float( output_file.read() )

        outputs['f_xy'] = f_xy


class TestExternalCodeFeature(unittest.TestCase):

    def tearDown(self):
        try:
            os.remove('paraboloid_input.dat')
        except OSError:
            pass
        try:
            os.remove('paraboloid_output.dat')
        except OSError:
            pass

    def test_main(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.components.tests.test_external_code import ParaboloidExternalCode

        top = Problem()
        top.model = model = Group()

        # Create and connect inputs
        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('p', ParaboloidExternalCode())

        model.connect('p1.x', 'p.x')
        model.connect('p2.y', 'p.y')

        # Run the ExternalCode Component
        top.setup()
        top.run_model()

        # Print the output
        self.assertEqual(top['p.f_xy'], -15.0)

if __name__ == "__main__":
    unittest.main()
