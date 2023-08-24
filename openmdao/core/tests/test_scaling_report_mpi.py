"""Define MPI version of the scaling report tests."""
import os
import unittest

import openmdao.core.tests.test_scaling_report as NonMPI


@unittest.skipIf(os.environ.get("GITHUB_ACTION"), "Unreliable on GitHub Actions workflows.")
class TestDriverScalingReportMPI(NonMPI.TestDriverScalingReport):
    N_PROCS = 2


@unittest.skipIf(os.environ.get("GITHUB_ACTION"), "Unreliable on GitHub Actions workflows.")
class TestDriverScalingReport2MPI(NonMPI.TestDriverScalingReport2):
    N_PROCS = 2


class TestDriverScalingReport3MPI(NonMPI.TestDriverScalingReport3):
    N_PROCS = 2


class TestDiscreteScalingReportMPI(NonMPI.TestDiscreteScalingReport):
    N_PROCS = 2


if __name__ == '__main__':
    unittest.main()
