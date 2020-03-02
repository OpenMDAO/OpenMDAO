import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI


try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


def _get_problem():
    p = om.Problem(model=om.Group())

    p.model.add_subsystem('des_vars', om.IndepVarComp('x', 150))

    sa = p.model.add_subsystem('G1', om.Group())
    sa.add_subsystem('C3', om.ExecComp('y = x', x=6))

    # the bug only appears if there is a NewtonSolver here...
    sa.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    par = p.model.add_subsystem('par', om.ParallelGroup())

    par.add_subsystem('C1', om.ExecComp('y = x'))
    p.model.connect('G1.C3.y', 'par.C1.x')

    par.add_subsystem('C2', om.ExecComp('y = x'))
    p.model.connect('G1.C3.y', 'par.C2.x')

    p.model.add_subsystem('con', om.ExecComp('y1=x1'))
    p.model.connect('des_vars.x', 'con.x1')

    # the bug only appears if force_alloc_complex is True
    p.setup(mode='auto', force_alloc_complex=True)

    p.set_solver_print(level=0)

    p.run_model()

    return p


class SerialTestCase(unittest.TestCase):

    def test_serial_model(self):
        p = _get_problem()
        J = p.compute_totals(of=['con.y1'], wrt=['des_vars.x'], return_format='array')
        np.testing.assert_allclose(J, np.array([[1.0]]), rtol=1e-7, atol=0, equal_nan=True,
                                   err_msg='', verbose=True)

@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParallelTestCase(unittest.TestCase):

    N_PROCS = 2

    def test_mpi_model(self):
        p = _get_problem()
        J = p.compute_totals(of=['con.y1'], wrt=['des_vars.x'], return_format='array')
        np.testing.assert_allclose(J, np.array([[1.0]]), rtol=1e-7, atol=0, equal_nan=True,
                                   err_msg='', verbose=True)

if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
