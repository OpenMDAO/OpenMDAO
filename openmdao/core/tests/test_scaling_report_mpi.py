"""Define the units/scaling tests."""
import unittest

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.core.tests.test_scaling_report import TestDriverScalingReport


class TestDriverScalingReportMPI(TestDriverScalingReport):
    N_PROCS = 2


if __name__ == '__main__':
    unittest.main()
