import unittest
import os
import sys
from subprocess import Popen, PIPE, STDOUT

class LintTestCase(unittest.TestCase):
    # TODO: instead of running both together, we could split up
    # the pep8 and pep257 checks into two different tests so we could run
    # them concurrently.
    def test_pep8_pep257(self):
        tdir = os.path.dirname(os.path.abspath(__file__))
        p = Popen([sys.executable, 'openmdao_lint.py'],
                  stdout=PIPE, stderr=STDOUT, env=os.environ, cwd=tdir)
        output = p.communicate()[0]

        msgs = [line for line in output.split('\n') if ':' in line]

        if p.returncode:
            self.fail('pep8/pep257 failure: %s' % '\n'.join(msgs))
