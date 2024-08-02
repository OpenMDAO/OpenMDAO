import unittest

from contextlib import redirect_stdout, contextmanager
import io
import os
import sys

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs


@contextmanager
def _replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


@use_tempdirs
class TestCleanOutputs(unittest.TestCase):

    def test_specify_prob(self):

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
            om.clean_outputs(p1, dryrun=True)

        expected = ('Found 1 OpenMDAO output directories:\n'
                    '  foo_out\n'
                    'Would remove 1 output directories (dryrun = True).\n')
        
        self.assertIn(expected, ss.getvalue())

        # Now actually do it.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(p1)

        self.assertNotIn('foo_out', os.listdir(os.getcwd()))
        self.assertIn('bar_out', os.listdir(os.getcwd()))

        # Now specify p2 with the output directory.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(p2)
        
        self.assertNotIn('bar_out', os.listdir(os.getcwd()))

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

        print(ss.getvalue())

        expected = ('Found 2 OpenMDAO output directories:\n'
                    '  bar_out\n'
                    '  foo_out\n'
                    'Would remove 2 output directories (dryrun = True).')
        
        self.assertIn(expected, ss.getvalue())

        # Test that no specified path gives the same result.
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(dryrun=True)

        expected = ('Found 2 OpenMDAO output directories:\n'
                    '  bar_out\n'
                    '  foo_out\n'
                    'Would remove 2 output directories (dryrun = True).')
        
        self.assertIn(expected, ss.getvalue())

        # Now remove the files
        ss = io.StringIO()
        with redirect_stdout(ss):
            om.clean_outputs(prompt=False)
        
        expected = ('Found 2 OpenMDAO output directories:\n'
                    '  bar_out\n'
                    '  foo_out\n'
                    'Removed 2 OpenMDAO output directories.\n')
        
        self.assertIn(expected, ss.getvalue())

        self.assertNotIn('foo_out', os.listdir(os.getcwd()))
        self.assertNotIn('bar_out', os.listdir(os.getcwd()))

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

                # First, respond in the negative
                ss = io.StringIO()
                with redirect_stdout(ss):
                    with _replace_stdin(io.StringIO('n')):
                        om.clean_outputs(recurse=recurse)
                
                if recurse:
                    expected = ('Found 2 OpenMDAO output directories:\n')
                    self.assertIn(expected, ss.getvalue())

                    outdirs = [d for d in os.listdir(os.getcwd()) if d.endswith('_out')]           
                    self.assertEqual(len(outdirs), 2)

                    # Respond in the positive to actually remove them.
                    ss = io.StringIO()
                    with redirect_stdout(ss):
                        with _replace_stdin(io.StringIO('y')):
                            om.clean_outputs()

                    expected = ('Removed 2 OpenMDAO output directories.\n')
                    
                    self.assertIn(expected, ss.getvalue())

                    outdirs = [d for d in os.listdir(os.getcwd()) if d.endswith('_out')]           
                    self.assertEqual(len(outdirs), 0)
                else:
                    print(ss.getvalue())
