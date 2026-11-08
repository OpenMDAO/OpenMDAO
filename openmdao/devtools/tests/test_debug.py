import unittest

from io import StringIO

from openmdao.api import ExecComp, ParallelGroup, Problem
from openmdao.test_suite.scripts.circle_opt import CircleOpt
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.devtools.debug import config_summary

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


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


@unittest.skipUnless(MPI and PETScVector, 'MPI and PETSc are required.')
@use_tempdirs
class TestDebugMPI(unittest.TestCase):

    N_PROCS = 2

    def test_summary(self):
        prob = Problem()
        par = prob.model.add_subsystem('par', ParallelGroup())
        par.add_subsystem('c1', ExecComp('y = 2.0 * x'))
        par.add_subsystem('c2', ExecComp('y = 3.0 * x'))
        prob.setup()
        prob.final_setup()

        stdout = StringIO()
        config_summary(prob, stream=stdout)

        if prob.comm.rank == 0:
            self.assertIn('Max tree depth:       2', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
