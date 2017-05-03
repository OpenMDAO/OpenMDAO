""" Test out some crucial linear GS tests in parallel with distributed comps."""

import unittest
import numpy

from openmdao.api import ParallelGroup, Group, Problem, IndepVarComp, \
    ExecComp, LinearBlockGS
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
    An ExecComp that uses 2 procs and takes input var slices.
    """
    def __init__(self, exprs, arr_size=11, **kwargs):
        super(DistribExecComp, self).__init__(exprs, **kwargs)
        self.arr_size = arr_size

    def initialize_variables(self):
        outs = set()
        allvars = set()
        exprs = self._exprs
        kwargs = self._kwargs

        # find all of the variables and which ones are outputs
        for expr in exprs:
            lhs, _ = expr.split('=', 1)
            outs.update(self._parse_for_out_vars(lhs))
            allvars.update(self._parse_for_vars(expr))

        comm = self.comm
        rank = comm.rank

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

        super(DistribExecComp, self).initialize_variables()

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
        prob.model.ln_solver = LinearBlockGS()
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

    def test_two_simple(self):
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
        prob.model.ln_solver = LinearBlockGS()
        prob.model.connect('P.x', 'C1.x')
        prob.model.connect('C1.y', 'C2.y')

        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.run_model()

        J = prob.compute_total_derivs(['C2.z'], ['P.x'])
        assert_rel_error(self, J['C2.z', 'P.x'], numpy.eye(size)*6.0, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(['C2.z'], ['P.x'])
        assert_rel_error(self, J['C2.z', 'P.x'], numpy.eye(size)*6.0, 1e-6)

    def test_fan_out_grouped(self):
        size = 3
        prob = Problem()
        prob.model = root = Group()
        root.add_subsystem('P', IndepVarComp('x', numpy.ones(size, dtype=float)))
        root.add_subsystem('C1', DistribExecComp(['y=3.0*x'], arr_size=size,
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

        assert_rel_error(self, prob['C2.y'], numpy.ones(size)*4.5)
        assert_rel_error(self, prob['C3.y'], numpy.ones(size)*15.0)

        J = prob.compute_total_derivs(of=['C2.y', "C3.y"], wrt=['P.x'])
        assert_rel_error(self, J['C2.y', 'P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y', 'P.x'], numpy.eye(size)*15.0, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=['C2.y', "C3.y"], wrt=['P.x'])
        assert_rel_error(self, J['C2.y', 'P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y', 'P.x'], numpy.eye(size)*15.0, 1e-6)

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
        root.add_subsystem('C3', DistribExecComp(['y=3.0*x1+7.0*x2'], arr_size=size,
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

        root.ln_solver = LinearBlockGS()
        sub.ln_solver = LinearBlockGS()

        prob.model.suppress_solver_output = True
        sub.suppress_solver_output = True
        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_total_derivs(of=['C4.y'], wrt=['P1.x', 'P2.x'])
        assert_rel_error(self, J['C4.y', 'P1.x'], numpy.eye(size)*-6.0, 1e-6)
        assert_rel_error(self, J['C4.y', 'P2.x'], numpy.eye(size)*35.0, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')

        prob.run_driver()

        J = prob.compute_total_derivs(of=['C4.y'], wrt=['P1.x', 'P2.x'])
        assert_rel_error(self, J['C4.y', 'P1.x'], numpy.eye(size)*-6.0, 1e-6)
        assert_rel_error(self, J['C4.y', 'P2.x'], numpy.eye(size)*35.0, 1e-6)

    def test_src_indices_error(self):
        raise unittest.SkipTest("figure out API for determining distributed vars first")
        size = 3
        group = Group()
        P = group.add_subsystem('P', IndepVarComp('x', numpy.ones(size)))
        C1 = group.add_subsystem('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                                       x=numpy.zeros(size),
                                                       y=numpy.zeros(size)))
        C2 = group.add_subsystem('C2', ExecComp(['z=3.0*y'],
                                                y=numpy.zeros(size),
                                                z=numpy.zeros(size)))

        prob = Problem()
        prob.model = group
        prob.model.ln_solver = LinearBlockGS()
        prob.model.connect('P.x', 'C1.x')
        prob.model.connect('C1.y', 'C2.y')

        P.add_design_var('x', lower=0., upper=100.)
        C1.add_objective('y')

        try:
            prob.setup(vector_class=PETScVector, check=False)
        except Exception as err:
            self.assertEqual(str(err), "'C1.y' is a distributed variable"
                                       " and may not be used as a design var,"
                                       " objective, or constraint.")
        else:
            if MPI:
                self.fail("Exception expected")


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
