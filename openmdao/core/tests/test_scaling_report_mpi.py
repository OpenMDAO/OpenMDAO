"""Define the units/scaling tests."""
import unittest

import  openmdao.core.tests.test_scaling_report as NonMPI


class TestDriverScalingReportMPI(NonMPI.TestDriverScalingReport):
    N_PROCS = 2


if __name__ == '__main__':
    unittest.main()
