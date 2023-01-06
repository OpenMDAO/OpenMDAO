
import unittest
import numpy as np

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.mpi import MPI, multi_proc_exception_check
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.core.tests.test_coloring import build_multipoint_problem

OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


@use_tempdirs
@unittest.skipUnless(MPI and PETScVector and OPTIMIZER, "MPI, PETSc, and pyOptSparse are required.")
class MatMultMultipointMPI2TestCase(unittest.TestCase):
    N_PROCS = 2

    def test_multipoint_with_coloring(self):
        num_pts = 4
        p = build_multipoint_problem(size=10, num_pts=num_pts)
        p.setup()

        p.run_driver()

        with multi_proc_exception_check(p.comm):
            J = p.compute_totals()

        for i in range(num_pts):
            with multi_proc_exception_check(p.comm):
                A1 = p.get_val('par1.comp%d.A'%i, get_remote=True)
            with multi_proc_exception_check(p.comm):
                A2 = p.get_val('par2.comp%d.A'%i, get_remote=True)
            norm = np.linalg.norm(J['par2.comp%d.y'%i,'indep%d.x'%i] - A2.dot(A1))
            with multi_proc_exception_check(p.comm):
                self.assertLess(norm, 1.e-7)

        with multi_proc_exception_check(p.comm):
            print("final obj:", p['obj.y'])


if __name__ == '__main__':
    unittest.main()
