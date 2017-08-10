""" Test out some crucial linear GS tests in parallel with distributed comps."""

import unittest
import numpy

from openmdao.api import ParallelGroup, Group, Problem, IndepVarComp, \
    ExecComp, LinearBlockGS, ExplicitComponent, ImplicitComponent, PetscKSP
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.devtools.testutil import assert_rel_error

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

if MPI:
    rank = MPI.COMM_WORLD.rank
else:
    rank = 0

class DistribExecComp(ExecComp):
    """
    An ExecComp that uses N procs and takes input var slices.  Unlike a normal
    ExecComp, if only supports a single expression per proc.  If you give it
    multiple expressions, it will use a different one in each proc, repeating
    the last one in any remaining procs.
    """
    def __init__(self, exprs, arr_size=11, **kwargs):
        super(DistribExecComp, self).__init__(exprs, **kwargs)
        self.arr_size = arr_size
        self.distributed = True

    def setup(self):
        outs = set()
        allvars = set()
        exprs = self._exprs
        kwargs = self._kwargs

        comm = self.comm
        rank = comm.rank

        if len(self._exprs) > comm.size:
            raise RuntimeError("DistribExecComp only supports up to 1 expression per MPI process.")

        if len(self._exprs) < comm.size:
            # repeat the last expression for any leftover procs
            self._exprs.extend([self._exprs[-1]] * (comm.size - len(self._exprs)))

        self._exprs = [self._exprs[rank]]

        # find all of the variables and which ones are outputs
        for expr in exprs:
            lhs, _ = expr.split('=', 1)
            outs.update(self._parse_for_out_vars(lhs))
            allvars.update(self._parse_for_vars(expr))

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        for name in outs:
            if name not in kwargs or not isinstance(kwargs[name], dict):
                kwargs[name] = {}
            kwargs[name]['value'] = numpy.ones(sizes[rank], float)

        for name in allvars:
            if name not in outs:
                if name not in kwargs or not isinstance(kwargs[name], dict):
                    kwargs[name] = {}
                meta = kwargs[name]
                meta['value'] = numpy.ones(sizes[rank], float)
                meta['src_indices'] = numpy.arange(start, end, dtype=int)

        super(DistribExecComp, self).setup()

    def get_req_procs(self):
        return (2, None)


class DistribCoordComp(ExplicitComponent):
    def __init__(self, **kwargs):
        super(DistribCoordComp, self).__init__(**kwargs)
        self.distributed = True

    def setup(self):
        comm = self.comm
        rank = comm.rank

        if rank == 0:
            self.add_input('invec', numpy.zeros((5, 3)),
                           src_indices=[[(0,0), (0,1), (0,2)],
                                        [(1,0), (1,1), (1,2)],
                                        [(2,0), (2,1), (2,2)],
                                        [(3,0), (3,1), (3,2)],
                                        [(4,0), (4,1), (4,2)]])
            self.add_output('outvec', numpy.zeros((5, 3)))
        else:
            self.add_input('invec', numpy.zeros((4, 3)),
                           src_indices=[[(5,0), (5,1), (5,2)],
                                        [(6,0), (6,1), (6,2)],
                                        [(7,0), (7,1), (7,2)],
                                        # use some negative indices here to
                                        # make sure they work
                                        [(-1,0), (8,1), (-1,2)]])
            self.add_output('outvec', numpy.zeros((4, 3)))

    def compute(self, inputs, outputs):
        if self.comm.rank == 0:
            outputs['outvec'] = inputs['invec'] * 2.0
        else:
            outputs['outvec'] = inputs['invec'] * 3.0

    def get_req_procs(self):
        return (2, 2)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITests1(unittest.TestCase):

    N_PROCS = 1

    def test_too_few_procs(self):
        size = 3
        group = Group()
        group.add_subsystem('P', IndepVarComp('x', numpy.ones(size)))
        group.add_subsystem('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                        x=numpy.zeros(size),
                                        y=numpy.zeros(size)))
        group.add_subsystem('C2', ExecComp(['z=3.0*y'],
                                 y=numpy.zeros(size),
                                 z=numpy.zeros(size)))

        prob = Problem()
        prob.model = group
        prob.model.linear_solver = LinearBlockGS()
        prob.model.connect('P.x', 'C1.x')
        prob.model.connect('C1.y', 'C2.y')

        try:
            prob.setup(vector_class=PETScVector, check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "C1 needs 2 MPI processes, but was given only 1.")
        else:
            if MPI:
                self.fail("Exception expected")


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITests2(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_shape(self):
        points = numpy.array([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],

            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.],
            [0., 0., 2.],
        ])

        prob = Problem()

        prob.model.add_subsystem('indep', IndepVarComp('x', points))
        prob.model.add_subsystem('comp', DistribCoordComp())
        prob.model.add_subsystem('total', ExecComp('y=x',
                                                   x=numpy.zeros((9,3)),
                                                   y=numpy.zeros((9,3))))
        prob.model.connect('indep.x', 'comp.invec')
        prob.model.connect('comp.outvec', 'total.x')

        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.run_model()

        final = points.copy()
        final[0:5] *= 2.0
        final[5:9] *= 3.0

        assert_rel_error(self, prob['total.y'], final)

    def test_two_simple(self):
        size = 3
        group = Group()

        # import pydevd
        # pydevd.settrace('localhost', port=10000+MPI.COMM_WORLD.rank,
        #                 stdoutToServer=True, stderrToServer=True)

        group.add_subsystem('P', IndepVarComp('x', numpy.arange(size)))
        group.add_subsystem('C1', DistribExecComp(['y=2.0*x', 'y=3.0*x'], arr_size=size,
                                                  x=numpy.zeros(size),
                                                  y=numpy.zeros(size)))
        group.add_subsystem('C2', ExecComp(['z=3.0*y'],
                                           y=numpy.zeros(size),
                                           z=numpy.zeros(size)))

        prob = Problem()
        prob.model = group
        prob.model.linear_solver = LinearBlockGS()
        prob.model.connect('P.x', 'C1.x')
        prob.model.connect('C1.y', 'C2.y')

        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.run_model()

        J = prob.compute_total_derivs(['C2.z'], ['P.x'])
        assert_rel_error(self, J['C2.z', 'P.x'], numpy.diag([6.0, 6.0, 9.0]), 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(['C2.z'], ['P.x'])
        assert_rel_error(self, J['C2.z', 'P.x'], numpy.diag([6.0, 6.0, 9.0]), 1e-6)

    def test_fan_out_grouped(self):
        size = 3
        prob = Problem()
        prob.model = root = Group()
        root.add_subsystem('P', IndepVarComp('x', numpy.ones(size, dtype=float)))
        root.add_subsystem('C1', DistribExecComp(['y=3.0*x', 'y=2.0*x'], arr_size=size,
                                                 x=numpy.zeros(size, dtype=float),
                                                 y=numpy.zeros(size, dtype=float)))
        sub = root.add_subsystem('sub', ParallelGroup())
        sub.add_subsystem('C2', ExecComp('y=1.5*x',
                                         x=numpy.zeros(size),
                                         y=numpy.zeros(size)))
        sub.add_subsystem('C3', ExecComp(['y=5.0*x'],
                                         x=numpy.zeros(size, dtype=float),
                                         y=numpy.zeros(size, dtype=float)))

        root.add_subsystem('C2', ExecComp(['y=x'],
                                          x=numpy.zeros(size, dtype=float),
                                          y=numpy.zeros(size, dtype=float)))
        root.add_subsystem('C3', ExecComp(['y=x'],
                                          x=numpy.zeros(size, dtype=float),
                                          y=numpy.zeros(size, dtype=float)))
        root.connect('sub.C2.y', 'C2.x')
        root.connect('sub.C3.y', 'C3.x')

        root.connect("C1.y", "sub.C2.x")
        root.connect("C1.y", "sub.C3.x")
        root.connect("P.x", "C1.x")

        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.run_model()

        diag1 = [4.5, 4.5, 3.0]
        diag2 = [15.0, 15.0, 10.0]

        assert_rel_error(self, prob['C2.y'], diag1)
        assert_rel_error(self, prob['C3.y'], diag2)

        diag1 = numpy.diag(diag1)
        diag2 = numpy.diag(diag2)

        J = prob.compute_total_derivs(of=['C2.y', "C3.y"], wrt=['P.x'])
        assert_rel_error(self, J['C2.y', 'P.x'], diag1, 1e-6)
        assert_rel_error(self, J['C3.y', 'P.x'], diag2, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=['C2.y', "C3.y"], wrt=['P.x'])
        assert_rel_error(self, J['C2.y', 'P.x'], diag1, 1e-6)
        assert_rel_error(self, J['C3.y', 'P.x'], diag2, 1e-6)

    def test_fan_in_grouped(self):
        size = 3

        prob = Problem()
        prob.model = root = Group()

        root.add_subsystem('P1', IndepVarComp('x', numpy.ones(size, dtype=float)))
        root.add_subsystem('P2', IndepVarComp('x', numpy.ones(size, dtype=float)))
        sub = root.add_subsystem('sub', ParallelGroup())

        sub.add_subsystem('C1', ExecComp(['y=-2.0*x'],
                                         x=numpy.zeros(size, dtype=float),
                                         y=numpy.zeros(size, dtype=float)))
        sub.add_subsystem('C2', ExecComp(['y=5.0*x'],
                                         x=numpy.zeros(size, dtype=float),
                                         y=numpy.zeros(size, dtype=float)))
        root.add_subsystem('C3', DistribExecComp(['y=3.0*x1+7.0*x2', 'y=1.5*x1+3.5*x2'], arr_size=size,
                                                 x1=numpy.zeros(size, dtype=float),
                                                 x2=numpy.zeros(size, dtype=float),
                                                 y=numpy.zeros(size, dtype=float)))
        root.add_subsystem('C4', ExecComp(['y=x'],
                                          x=numpy.zeros(size, dtype=float),
                                          y=numpy.zeros(size, dtype=float)))

        root.connect("sub.C1.y", "C3.x1")
        root.connect("sub.C2.y", "C3.x2")
        root.connect("P1.x", "sub.C1.x")
        root.connect("P2.x", "sub.C2.x")
        root.connect("C3.y", "C4.x")

        root.linear_solver = LinearBlockGS()
        sub.linear_solver = LinearBlockGS()

        prob.model.suppress_solver_output = True
        sub.suppress_solver_output = True
        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.run_driver()

        diag1 = numpy.diag([-6.0, -6.0, -3.0])
        diag2 = numpy.diag([35.0, 35.0, 17.5])

        J = prob.compute_total_derivs(of=['C4.y'], wrt=['P1.x', 'P2.x'])
        assert_rel_error(self, J['C4.y', 'P1.x'], diag1, 1e-6)
        assert_rel_error(self, J['C4.y', 'P2.x'], diag2, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')

        prob.run_driver()

        J = prob.compute_total_derivs(of=['C4.y'], wrt=['P1.x', 'P2.x'])
        assert_rel_error(self, J['C4.y', 'P1.x'], diag1, 1e-6)
        assert_rel_error(self, J['C4.y', 'P2.x'], diag2, 1e-6)

    def test_distrib_voi(self):
        raise unittest.SkipTest("distrib vois no supported yet")



class DistribStateImplicit(ImplicitComponent):
    """
    This component is unusual in that it has a distributed variable 'states' that
    is not connected to any other variables in the model.  The input 'a' sets the local
    values of 'states' and the output 'out_var' is the sum of all of the distributed values
    of 'states'.
    """
    def setup(self):
        self.add_input('a', val=10., units='m')

        rank = self.comm.rank

        GLOBAL_SIZE = 5
        sizes, offsets = evenly_distrib_idxs(self.comm.size, GLOBAL_SIZE)

        self.add_output('states', shape=int(sizes[rank]))

        self.add_output('out_var', shape=1)

        self.local_size = sizes[rank]

        self.linear_solver = PetscKSP()

    def get_req_procs(self):
        return 1,10

    def solve_nonlinear(self, i, o):
        o['states'] = i['a']

        local_sum = numpy.zeros(1)
        local_sum[0] = numpy.sum(o['states'])
        tmp = numpy.zeros(1)
        self.comm.Allreduce(local_sum, tmp, op=MPI.SUM)

        o['out_var'] = tmp[0]

    def apply_nonlinear(self, i, o, r):
        r['states'] = o['states'] - i['a']

        local_sum = numpy.zeros(1)
        local_sum[0] = numpy.sum(o['states'])
        global_sum = numpy.zeros(1)
        self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)

        r['out_var'] = o['out_var'] - global_sum[0]

    def apply_linear(self, i, o, d_i, d_o, d_r, mode):
        if mode == 'fwd':
            if 'states' in d_o:
                d_r['states'] += d_o['states']

                local_sum = numpy.array([numpy.sum(d_o['states'])])
                global_sum = numpy.zeros(1)
                self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
                d_r['out_var'] -= global_sum

            if 'out_var' in d_o:
                    d_r['out_var'] += d_o['out_var']

            if 'a' in d_i:
                    d_r['states'] -= d_i['a']

        elif mode == 'rev':
            if 'states' in d_o:
                d_o['states'] += d_r['states']

                tmp = numpy.zeros(1)
                if self.comm.rank == 0:
                    tmp[0] = d_r['out_var'].copy()
                self.comm.Bcast(tmp, root=0)

                d_o['states'] -= tmp

            if 'out_var' in d_o:
                d_o['out_var'] += d_r['out_var']

            if 'a' in d_i:
                    d_i['a'] -= numpy.sum(d_r['states'])

@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITests3(unittest.TestCase):

    N_PROCS = 3

    def test_distrib_apply(self):
        p = Problem()

        p.model.add_subsystem('des_vars', IndepVarComp('a', val=10., units='m'), promotes=['*'])
        p.model.add_subsystem('icomp', DistribStateImplicit(), promotes=['*'])

        expected = numpy.array([[5.]])

        p.setup(vector_class=PETScVector, mode='fwd')
        p.run_model()
        jac = p.compute_total_derivs(of=['out_var'], wrt=['a'], return_format='dict')
        assert_rel_error(self, jac['out_var']['a'], expected, 1e-6)

        p.setup(vector_class=PETScVector, mode='rev')
        p.run_model()
        jac = p.compute_total_derivs(of=['out_var'], wrt=['a'], return_format='dict')
        assert_rel_error(self, jac['out_var']['a'], expected, 1e-6)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
