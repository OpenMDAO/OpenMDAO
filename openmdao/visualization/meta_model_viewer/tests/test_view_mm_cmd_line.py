""" Unit tests for structured metamodels in view_mm. """
import unittest
import subprocess
import os

try:
    import bokeh
except ImportError:
    bokeh = None
import openmdao.test_suite.test_examples.meta_model_examples.structured_meta_model_example as example

@unittest.skipUnless(bokeh, "Bokeh is required")
class ViewMMCommandLineTest(unittest.TestCase):

    def test_unspecified_metamodel(self):
        script = os.path.join(os.path.dirname(__file__), 'example.py')
        cmd = 'openmdao view_mm {}'.format(script)
        output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')
        expected_output = ("Metamodel not specified. "
                           "Try one of the following: ['interp1', 'interp2'].")
        self.assertTrue(
            expected_output in output,
            msg='Metamodel was specified when it should not have been. Check example.')

    def test_invalid_metamodel(self):
        script = os.path.abspath(example.__file__).replace('.pyc', '.py') # PY2
        cmd = 'openmdao view_mm {} -m {}'.format(script, 'interp')
        output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')
        expected_output = '\n'.join([
            "Metamodel 'interp' not found.",
            " Try one of the following: ['mm']."
        ])
        self.assertTrue(
            expected_output in output.replace('\r', ''),
            msg='Metamodel was found when it should not have. Check example.')
