from __future__ import print_function

import unittest

from openmdao.api import Problem, Group, ExecComp
from openmdao.api import Group, ParallelGroup, Problem, IndepVarComp, LinearBlockGS, DefaultVector, \
    ExecComp, ExplicitComponent, PETScVector, ScipyIterativeSolver, NonlinearBlockGS
from openmdao.utils.mpi import MPI

if MPI:
    from openmdao.api import PETScVector
    vector_class = PETScVector
else:
    vector_class = DefaultVector


def _build_model(nsubs, min_procs=None, max_procs=None, weights=None):
    p = Problem(model=ParallelGroup())
    if min_procs is None:
        min_procs = [1]*nsubs
    if max_procs is None:
        max_procs = [MPI.COMM_WORLD.size]*nsubs
    if weights is None:
        weights = [1.0]*nsubs

    model = p.model
    for i in range(nsubs):
        model.add_subsystem("C%d" % i, ExecComp("y=2.0*x"),
                            min_procs=min_procs[i], max_procs=max_procs[i], proc_weight=weights[i])

    p.setup(vector_class=vector_class, check=False)
    p.final_setup()

    return p

def _get_which_procs(group):
    sub_inds = [i for i, s in enumerate(group._subsystems_allprocs)
                if s in group._subsystems_myproc]
    return MPI.COMM_WORLD.allgather(sub_inds)

@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class ProcTestCase1(unittest.TestCase):

    N_PROCS = 1

    def test_proc(self):
        p = _build_model(nsubs=4)
        all_inds = _get_which_procs(p.model)
        self.assertEqual(all_inds, [[0,1,2,3]])

@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class ProcTestCase2(unittest.TestCase):

    N_PROCS = 2

    def test_proc(self):
        p = _build_model(nsubs=4)
        all_inds = _get_which_procs(p.model)
        self.assertEqual(all_inds, [[0,1],[2,3]])


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
