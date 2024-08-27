import unittest
import unittest.mock as mock

import builtins
from contextlib import redirect_stdout
import io
import os
import pathlib
import shutil
import sys

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.core.problem import _clear_problem_names


@use_tempdirs
class TestCleanOutputs(unittest.TestCase):

    def setUp(self):
        _clear_problem_names()

    def test_specify_prob(self):

        p1 = om.Problem()
        p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
        p1.setup()
        p1.run_model()

        p2 = om.Problem()
        p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
        p2.setup()
        p2.run_model()

        # First Test that a dryrun on p1 works as expected.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(p1, dryrun=True)

        expected1 = 'Removed 0 OpenMDAO output directories.\n'
        expected2 = 'Would remove'
        
        self.assertIn(expected1, ss.getvalue())
        self.assertIn(expected2, ss.getvalue())

        # Now actually do it.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(p1)

        self.assertEqual(len(os.listdir(os.getcwd())), 1)

        # Now specify p2 with the output directory.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(p2)
        
        self.assertEqual(len(os.listdir(os.getcwd())), 0)

    def test_specify_non_output_dir_no_prompt(self):

        p1 = om.Problem(name='foo')
        p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
        p1.setup()
        p1.run_model()

        p2 = om.Problem(name='bar')
        p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
        p2.setup()
        p2.run_model()

        # First Test that a dryrun on p1 works as expected.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs('.', dryrun=True)

        expected = ('Found 2 OpenMDAO output directories:',
                    'Would remove bar_out (dryrun = True).',
                    'Would remove foo_out (dryrun = True).',
                    'Removed 0 OpenMDAO output directories.')
        
        for expected_str in expected:
            self.assertIn(expected_str, ss.getvalue())

        # Test that no specified path gives the same result.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(dryrun=True)

        expected = ('Found 2 OpenMDAO output directories:',
                    'Would remove bar_out (dryrun = True).',
                    'Would remove foo_out (dryrun = True).',
                    'Removed 0 OpenMDAO output directories.')
        
        for expected_str in expected:
            self.assertIn(expected_str, ss.getvalue())

        # Now remove the files
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(prompt=False)
        
        expected = ('Found 2 OpenMDAO output directories:\n'
                    'Removed bar_out\n'
                    'Removed foo_out\n'
                    'Removed 2 OpenMDAO output directories.\n')
        
        self.assertIn(expected, ss.getvalue())

        self.assertNotIn('foo_out', os.listdir(os.getcwd()))
        self.assertNotIn('bar_out', os.listdir(os.getcwd()))

    @unittest.skipIf(sys.version_info  < (3, 9, 0), 'Requires Python 3.9.0 or later.')
    def test_specify_non_output_dir_prompt(self):

        for recurse in (True, False):
            with self.subTest(f'{recurse=}'):
                p1 = om.Problem()
                p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
                p1.setup()
                p1.run_model()

                p2 = om.Problem()
                p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
                p2.setup()
                p2.run_model()

                output_dirs = [p1.get_outputs_dir(), p2.get_outputs_dir()]

                pathlib.Path('temp').mkdir(exist_ok=True)
                for od in output_dirs:
                    shutil.move(od, 'temp')

                # First, respond in the negative
                ss = io.StringIO()
                with redirect_stdout(ss):
                    with mock.patch.object(builtins, 'input', lambda *_: 'n'):
                        om.clean_outputs(recurse=recurse)

                if recurse:
                    expected = ('Found 2 OpenMDAO output directories:\n')
                    self.assertIn(expected, ss.getvalue())

                    outdirs = [d for d in os.listdir('temp') if d.endswith('_out')]           
                    self.assertEqual(len(outdirs), 2)

                    # Respond in the positive to actually remove them.
                    ss = io.StringIO()
                    with redirect_stdout(ss):
                        with mock.patch.object(builtins, 'input', lambda *_: 'y'):
                            om.clean_outputs(recurse=recurse)

                    expected = ('Removed 2 OpenMDAO output directories.\n')
                    
                    self.assertIn(expected, ss.getvalue())

                    outdirs = [d for d in os.listdir('temp') if d.endswith('_out')]           
                    self.assertEqual(len(outdirs), 0)
                else:
                    self.assertIn('No OpenMDAO output directories found.', ss.getvalue())
                
                shutil.rmtree('temp')

    def test_pattern(self):

        p1 = om.Problem(name='foo')
        p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
        p1.setup()
        p1.run_model()

        p2 = om.Problem(name='bar')
        p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
        p2.setup()
        p2.run_model()

        # Make another non-openmdao outut directory and make sure we dont remove it.
        pathlib.Path('baz_out').mkdir(exist_ok=True)

        # First Test that a dryrun on p1 works as expected.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs('.', pattern='foo*', dryrun=True)

        expected = ('Found 1 OpenMDAO output directories:',
                    'Would remove foo_out (dryrun = True).',
                    'Removed 0 OpenMDAO output directories.')
        
        for expected_str in expected:
            self.assertIn(expected_str, ss.getvalue())

        # First Test that a dryrun on p1 works as expected.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs('.', pattern='*', dryrun=True)

        expected = ('Found 2 OpenMDAO output directories:',
                    'Would remove foo_out (dryrun = True).',
                    'Would remove bar_out (dryrun = True).',
                    'Removed 0 OpenMDAO output directories.')
        
        try:
            for expected_str in expected:
                self.assertIn(expected_str, ss.getvalue())
        finally:
            shutil.rmtree('baz_out')

    def test_recurse(self):

        p1 = om.Problem(name='foo')
        p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
        p1.setup()
        p1.run_model()

        p2 = om.Problem(name='bar')
        p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
        p2.setup()
        p2.run_model()

        # Make another non-openmdao outut directory to test recursion.
        pathlib.Path('baz_out').mkdir(exist_ok=True)
        shutil.move('foo_out', 'baz_out')
        shutil.move('bar_out', 'baz_out')

        # First Test that a dryrun on p1 works as expected.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs('.', pattern='*', dryrun=True, recurse=True)

        expected = ('Found 2 OpenMDAO output directories:',
                    f'Would remove baz_out{os.sep}foo_out (dryrun = True).',
                    f'Would remove baz_out{os.sep}bar_out (dryrun = True).',
                    'Removed 0 OpenMDAO output directories.')
        
        try:
            for expected_str in expected:
                self.assertIn(expected_str, ss.getvalue())
        finally:
            shutil.rmtree('baz_out')

    def test_norecurse(self):

        p1 = om.Problem(name='foo')
        p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
        p1.setup()
        p1.run_model()

        p2 = om.Problem(name='bar')
        p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
        p2.setup()
        p2.run_model()

        # Make another non-openmdao outut directory to test recursion.
        pathlib.Path('baz_out').mkdir(exist_ok=True)
        shutil.move('foo_out', 'baz_out')
        shutil.move('bar_out', 'baz_out')

        # First Test that a dryrun on p1 works as expected.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs('.', dryrun=True, recurse=False)

        expected = ('No OpenMDAO output directories found.',)

        try:
            for expected_str in expected:
                self.assertIn(expected_str, ss.getvalue())
        finally:
            shutil.rmtree('baz_out')

    def test_multiple_paths(self):
        p1 = om.Problem(name='foo')
        p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
        p1.setup()
        p1.run_model()

        p2 = om.Problem(name='bar')
        p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
        p2.setup()
        p2.run_model()

        # Make another non-openmdao outut directory to test recursion.
        pathlib.Path('baz_out').mkdir(exist_ok=True)
        shutil.move('foo_out', 'baz_out')
        shutil.move('bar_out', 'baz_out')

        # First Test that a dryrun on p1 works as expected.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(('baz_out/foo_out', 'baz_out/bar_out'),
                             dryrun=True, recurse=False)

        expected = ('Found 2 OpenMDAO output directories:',
                    f'Would remove baz_out{os.sep}foo_out (dryrun = True).',
                    f'Would remove baz_out{os.sep}bar_out (dryrun = True).',
                    'Removed 0 OpenMDAO output directories.')

        try:
            for expected_str in expected:
                self.assertIn(expected_str, ss.getvalue())
        finally:
            shutil.rmtree('baz_out')
