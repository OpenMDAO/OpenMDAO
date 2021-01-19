import unittest

from io import StringIO

from openmdao.api import Problem
from openmdao.test_suite.scripts.circle_opt import CircleOpt

from openmdao.devtools.debug import config_summary


class TestDebug(unittest.TestCase):

    def test_summary(self):
        prob = Problem(model=CircleOpt())
        prob.setup()
        prob.run_driver()

        stdout = StringIO()
        config_summary(prob, stream=stdout)
        text = stdout.getvalue().split('\n')

        expected = [
            "============== Problem Summary ============",
            "Groups:               1",
            "Components:           8",
            "Max tree depth:       1",
            "",
            "Design variables:            3   Total size:       21",
            "",
            "Nonlinear Constraints:       4   Total size:       21",
            "    equality:                2                     11",
            "    inequality:              2                     10",
            "",
            "Linear Constraints:          1   Total size:        1",
            "    equality:                1                      1",
            "    inequality:              0                      0",
            "",
            "Objectives:                  1   Total size:        1",
            "",
            "Input variables:            11   Total size:       82",
            "Output variables:           10   Total size:       77",
            "",
            "Total connections: 11   Total transfer data size: 82",
            "",
            "Driver type: Driver",
            "Linear Solvers: [LinearRunOnce]",
            "Nonlinear Solvers: [NonlinearRunOnce]"
        ]

        for i in range(len(expected)):
            self.assertEqual(text[i], expected[i],
                            '\nExpected: %s\nReceived: %s\n' % (expected[i], text[i]))


if __name__ == "__main__":
    unittest.main()
