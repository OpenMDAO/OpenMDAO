
import unittest

from openmdao.api import Problem, Group, ExecComp
from openmdao.api import Group, ParallelGroup, Problem, IndepVarComp, LinearBlockGS, \
    ExecComp, ExplicitComponent, PETScVector, ScipyKrylov, NonlinearBlockGS
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_near_equal

try:
    from openmdao.api import PETScVector
except:
    PETScVector = None


def _build_model(nsubs, min_procs=None, max_procs=None, weights=None, top=None, mode='fwd'):
    p = Problem()
    if min_procs is None:
        min_procs = [1]*nsubs
    if max_procs is None:
        max_procs = [None]*nsubs
    if weights is None:
        weights = [1.0]*nsubs

    model = p.model

    model.add_subsystem('indep', IndepVarComp('x', 1.0))
    par = model.add_subsystem('par', ParallelGroup())
    for i in range(nsubs):
        par.add_subsystem("C%d" % i, ExecComp("y=2.0*x"),
                          min_procs=min_procs[i], max_procs=max_procs[i], proc_weight=weights[i])
        model.connect('indep.x', 'par.C%d.x' % i)

    s_sum = '+'.join(['x%d' % i for i in range(nsubs)])
    model.add_subsystem('objective', ExecComp("y=%s" % s_sum))

    for i in range(nsubs):
        model.connect('par.C%d.y' % i, 'objective.x%d' % i)

    model.add_design_var('indep.x')
    model.add_objective('objective.y')

    p.setup(mode=mode, check=False)
    p.final_setup()

    return p


def _get_which_procs(group):
    sub_inds = [i for s, i in group._subsystems_allprocs.values()
                if s in group._subsystems_myproc]
    return MPI.COMM_WORLD.allgather(sub_inds)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProcTestCase2(unittest.TestCase):

    N_PROCS = 2

    def test_4_subs(self):
        p = _build_model(nsubs=4, mode='rev')
        all_inds = _get_which_procs(p.model.par)
        self.assertEqual(all_inds, [[0,1],[2,3]])

        p.run_model()
        assert_near_equal(p['objective.y'], 8.0)

        J = p.compute_totals(['objective.y'], ['indep.x'], return_format='dict')
        assert_near_equal(J['objective.y']['indep.x'][0][0], 8.0, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProcTestCase3(unittest.TestCase):

    N_PROCS = 3

    def test_3_subs(self):
        p = _build_model(nsubs=3)
        all_inds = _get_which_procs(p.model.par)
        self.assertEqual(all_inds, [[0], [1], [2]])

        p.run_model()

        assert_near_equal(p['objective.y'], 6.0)

        J = p.compute_totals(['objective.y'], ['indep.x'], return_format='dict')
        assert_near_equal(J['objective.y']['indep.x'][0][0], 6.0, 1e-6)

    def test_4_subs(self):
        p = _build_model(nsubs=4)
        all_inds = _get_which_procs(p.model.par)
        self.assertEqual(all_inds, [[0, 1], [2], [3]])

        p.run_model()
        assert_near_equal(p['objective.y'], 8.0)

        J = p.compute_totals(['objective.y'], ['indep.x'], return_format='dict')
        assert_near_equal(J['objective.y']['indep.x'][0][0], 8.0, 1e-6)

    def test_4_subs_max2(self):
        p = _build_model(nsubs=4, max_procs=[2,2,2,2])
        all_inds = _get_which_procs(p.model.par)
        self.assertEqual(all_inds, [[0, 1], [2], [3]])

        p.run_model()
        assert_near_equal(p['objective.y'], 8.0)

        J = p.compute_totals(['objective.y'], ['indep.x'], return_format='dict')
        assert_near_equal(J['objective.y']['indep.x'][0][0], 8.0, 1e-6)

    def test_4_subs_with_mins(self):
        try:
            p = _build_model(nsubs=4, min_procs=[1,2,2,1])
        except Exception as err:
            self.assertEqual(str(err), "'par' <class ParallelGroup>: MPI process allocation failed: can't meet min_procs required because the sum of the min procs required exceeds the procs allocated and the min procs required is > 1 for the following subsystems: ['C1', 'C2']")
        else:
            self.fail("Exception expected.")

    def test_4_subs_with_mins_serial_plus_2_par(self):
        p = Problem()
        model = p.model
        model.add_subsystem('indep', IndepVarComp('x', 1.0))
        par1 = model.add_subsystem('par1', ParallelGroup())
        par2 = model.add_subsystem('par2', ParallelGroup())

        par1.add_subsystem("C0", ExecComp("y=2.0*x"), min_procs=2)
        par1.add_subsystem("C1", ExecComp("y=2.0*x"), min_procs=1)
        par2.add_subsystem("C2", ExecComp("y=2.0*x"), min_procs=1)
        par2.add_subsystem("C3", ExecComp("y=2.0*x"), min_procs=2)

        s_sum = '+'.join(['x%d' % i for i in range(4)])
        model.add_subsystem('objective', ExecComp("y=%s" % s_sum))

        model.connect('indep.x', ['par1.C0.x', 'par1.C1.x', 'par2.C2.x', 'par2.C3.x'])
        for i in range(2):
            model.connect('par1.C%d.y' % i, 'objective.x%d' % i)
        for i in range(2,4):
            model.connect('par2.C%d.y' % i, 'objective.x%d' % i)

        model.add_design_var('indep.x')
        model.add_objective('objective.y')

        p.setup()
        p.final_setup()

        all_inds = _get_which_procs(p.model)
        self.assertEqual(all_inds, [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
        all_inds = _get_which_procs(p.model.par1)
        self.assertEqual(all_inds, [[0], [0], [1]])
        all_inds = _get_which_procs(p.model.par2)
        self.assertEqual(all_inds, [[0], [1], [1]])

        p.run_model()
        assert_near_equal(p['objective.y'], 8.0)

        J = p.compute_totals(['objective.y'], ['indep.x'], return_format='dict')
        assert_near_equal(J['objective.y']['indep.x'][0][0], 8.0, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProcTestCase5(unittest.TestCase):

    N_PROCS = 5

    def test_4_subs_max2(self):
        p = _build_model(nsubs=4, max_procs=[2,2,2,2])
        all_inds = _get_which_procs(p.model.par)
        self.assertEqual(all_inds, [[0], [0], [1], [2], [3]])

        p.run_model()
        assert_near_equal(p['objective.y'], 8.0)

        J = p.compute_totals(['objective.y'], ['indep.x'], return_format='dict')
        assert_near_equal(J['objective.y']['indep.x'][0][0], 8.0, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProcTestCase6(unittest.TestCase):

    N_PROCS = 6

    def test_3_subs_over_max(self):
        try:
            p = _build_model(nsubs=3, max_procs=[1, 2, 2])
        except Exception as err:
            self.assertEqual(str(err), "'par' <class ParallelGroup>: too many MPI procs allocated. Comm is size 6 but can only use 5.")
        else:
            self.fail("Exception expected.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProcTestCase8(unittest.TestCase):

    N_PROCS = 8

    def test_4_subs_weighted(self):
        p = _build_model(nsubs=4, weights=[1.0, 2.0, 1.0, 4.0], mode='rev')
        all_inds = _get_which_procs(p.model.par)
        self.assertEqual(all_inds, [[0], [1], [1], [2], [3], [3], [3], [3]])

        p.run_model()
        assert_near_equal(p['objective.y'], 8.0)

        J = p.compute_totals(['objective.y'], ['indep.x'], return_format='dict')
        assert_near_equal(J['objective.y']['indep.x'][0][0], 8.0, 1e-6)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
