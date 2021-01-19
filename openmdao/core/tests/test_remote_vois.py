
import unittest
import numpy as np
import time
import random

import openmdao.api as om

from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_near_equal

if MPI:
    from openmdao.api import PETScVector
    vector_class = PETScVector
    try:
        from pyoptsparse import Optimization as pyoptsparse_opt
    except ImportError:
        pyoptsparse_opt = None

else:
    PETScVector = None
    pyoptsparse_opt = None


class Mygroup(om.Group):

    def setup(self):
        self.add_subsystem('indep_var_comp', om.IndepVarComp('x'), promotes=['*'])
        self.add_subsystem('Cy', om.ExecComp('y=2*x'), promotes=['*'])
        self.add_subsystem('Cc', om.ExecComp('c=x+2'), promotes=['*'])

        self.add_design_var('x')
        self.add_constraint('c', lower=-3.)


@unittest.skipUnless(MPI and PETScVector and pyoptsparse_opt,
                     "MPI, PETSc and pyoptsparse are required.")
class RemoteVOITestCase(unittest.TestCase):

    N_PROCS = 2

    def test_remote_voi(self):
        prob = om.Problem()

        par = prob.model.add_subsystem('par', om.ParallelGroup())

        prob.model.par.add_subsystem('G1', Mygroup())
        prob.model.par.add_subsystem('G2', Mygroup())

        prob.model.add_subsystem('Obj', om.ExecComp('obj=y1+y2'))

        prob.model.connect('par.G1.y', 'Obj.y1')
        prob.model.connect('par.G2.y', 'Obj.y2')

        prob.model.add_objective('Obj.obj')

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.setup()

        self.assertEqual('par.G1', par.G1.pathname)
        self.assertEqual('par.G2', par.G2.pathname)

        prob.run_driver()

        J = prob.compute_totals(of=['Obj.obj', 'par.G1.c', 'par.G2.c'],
                                wrt=['par.G1.x', 'par.G2.x'])

        assert_near_equal(J['Obj.obj', 'par.G1.x'], np.array([[2.0]]), 1e-6)
        assert_near_equal(J['Obj.obj', 'par.G2.x'], np.array([[2.0]]), 1e-6)
        assert_near_equal(J['par.G1.c', 'par.G1.x'], np.array([[1.0]]), 1e-6)
        assert_near_equal(J['par.G1.c', 'par.G2.x'], np.array([[0.0]]), 1e-6)
        assert_near_equal(J['par.G2.c', 'par.G1.x'], np.array([[0.0]]), 1e-6)
        assert_near_equal(J['par.G2.c', 'par.G2.x'], np.array([[1.0]]), 1e-6)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
