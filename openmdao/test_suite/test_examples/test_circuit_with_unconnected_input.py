import unittest
import subprocess
import os

import openmdao.test_suite.scripts


class TestCircuitWithUnconnectedInputScript(unittest.TestCase):

    def _run_command(self, cmd):
        try:
            output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')
        except subprocess.CalledProcessError as err:
            msg = "Running command '{}' failed. " + \
                  "Output was: \n{}".format(cmd, err.output.decode('utf-8'))
            self.fail(msg)

        return output

    def test_circuit_with_unconnected_input(self):

        script_path = os.path.join(os.path.dirname(openmdao.test_suite.scripts.__file__),
                              'circuit_with_unconnected_input.py')

        output = self._run_command('python {}'.format(script_path))

        self.assertTrue('The following inputs are not connected' in output,
                        msg="Should have gotten error about unconnected input")


if __name__ == '__main__':
    unittest.main()
