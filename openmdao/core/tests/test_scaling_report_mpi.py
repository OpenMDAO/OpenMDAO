"""Define MPI version of the scaling report tests."""
import unittest

import  openmdao.core.tests.test_scaling_report as NonMPI


class TestDriverScalingReportMPI(NonMPI.TestDriverScalingReport):
    N_PROCS = 2


class TestDriverScalingReport2MPI(NonMPI.TestDriverScalingReport2):
    N_PROCS = 2


class TestDriverScalingReport3MPI(NonMPI.TestDriverScalingReport3):
    N_PROCS = 2


class TestDiscreteScalingReportMPI(NonMPI.TestDiscreteScalingReport):
    N_PROCS = 2


if __name__ == '__main__':
    unittest.main()
