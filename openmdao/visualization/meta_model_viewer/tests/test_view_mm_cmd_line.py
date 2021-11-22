""" Unit tests for structured metamodels in view_mm. """
import unittest
import subprocess
import os

try:
    import bokeh
except ImportError:
    bokeh = None

@unittest.skipUnless(bokeh, "Bokeh is required")
class ViewMMCommandLineTest(unittest.TestCase):

    def test_unspecified_metamodel(self):
        script = os.path.join(os.path.dirname(__file__), 'meta_model_cli_example.py')
        cmd = 'openmdao view_mm {}'.format(script)
        output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')    # nosec: trusted input
        expected_output = ('Metamodel not specified. Try one of the following:'
                           'openmdao view_mm -m interp1 {}'
                           'openmdao view_mm -m interp2 {}'.format(script, script))
        self.assertTrue(expected_output, output)

    def test_invalid_metamodel(self):
        script = os.path.join(os.path.dirname(__file__), 'meta_model_cli_example.py')
        cmd = 'openmdao view_mm {} -m {}'.format(script, 'IndepMeta')
        output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')    # nosec: trusted input
        expected_output = (
            "Metamodel 'IndepMeta' not found. Try one of the following:"
            "openmdao view_mm -m interp1 {}"
            "openmdao view_mm -m interp2 {}".format(script, script)
        )
        self.assertTrue(expected_output, output)

    def test_not_metamodel(self):
        script = os.path.join(os.path.dirname(__file__), 'meta_model_cli_example.py')
        cmd = 'openmdao view_mm {} -m {}'.format(script, 'dummy')
        output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')    # nosec: trusted input
        expected_output = (
            "'dummy' is not a Metamodel. Try one of the following:"
            "openmdao view_mm -m interp1 {}"
            "openmdao view_mm -m interp2 {}".format(script, script)
        )
        self.assertTrue(expected_output, output)

    def test_single_meta_model_fail(self):
        script = os.path.join(os.path.dirname(__file__), 'single_meta_model_example.py')
        cmd = 'openmdao view_mm {} -m {}'.format(script, 'dummy')
        output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')    # nosec: trusted input
        expected_output = (
            "'dummy' is not a Metamodel. Try the following:"
            "openmdao view_mm -m interp1 {}".format(script)
        )
        self.assertTrue(expected_output, output)

    def test_invalid_single_metamodel(self):
        script = os.path.join(os.path.dirname(__file__), 'single_meta_model_example.py')
        cmd = 'openmdao view_mm {} -m {}'.format(script, 'IndepMeta')
        output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')    # nosec: trusted input
        expected_output = (
            "Metamodel 'IndepMeta' not found. Try the following:"
            "openmdao view_mm -m interp1 {}".format(script)
        )
        self.assertTrue(expected_output, output)