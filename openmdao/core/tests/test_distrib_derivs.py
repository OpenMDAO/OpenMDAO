""" Test out some crucial linear GS tests in parallel with distributed comps."""

from __future__ import print_function

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
    """An ExecComp that uses 2 procs and
    takes input var slices and has output var slices as well.
    """
    def __init__(self, exprs, arr_size=11, **kwargs):
        super(DistribExecComp, self).__init__(exprs, **kwargs)
        self.arr_size = arr_size

    def initialize_variables(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs. Returns a dict of
        index arrays keyed to variable names.
        """
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

        for name, meta in kwargs.items():
            if not isinstance(meta, dict):
                kwargs[name] = {}
            if name in outs:
                pass
            else:  # input
                kwargs[name]...
        for n, m in self._init_unknowns_dict.items():
            self.set_var_indices(n, val=numpy.ones(sizes[rank], float),
                                 src_indices=numpy.arange(start, end, dtype=int))

        for n, m in self._init_params_dict.items():
            self.set_var_indices(n, val=numpy.ones(sizes[rank], float),
                                 src_indices=numpy.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class MPITests1(unittest.TestCase):

    N_PROCS = 1

    def test_too_few_procs(self):
        size = 3
        group = Group()
        group.add('P', IndepVarComp('x', numpy.ones(size)))
        group.add('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                        x=numpy.zeros(size),
                                        y=numpy.zeros(size)))
        group.add('C2', ExecComp(['z=3.0*y'],
                                 y=numpy.zeros(size),
                                 z=numpy.zeros(size)))

        prob = Problem()
        prob.model = group
        prob.model.ln_solver = LinearBlockGS()
        prob.model.connect('P.x', 'C1.x')
        prob.model.connect('C1.y', 'C2.y')

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "This problem was given 1 MPI processes, "
                             "but it requires between 2 and 2.")
        else:
            if MPI:
                self.fail("Exception expected")


class MPITests2(unittest.TestCase):

    N_PROCS = 2

    def test_two_simple(self):
        size = 3
        group = Group()
        group.add('P', IndepVarComp('x', numpy.ones(size)))
        group.add('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                        x=numpy.zeros(size),
                                        y=numpy.zeros(size)))
        group.add('C2', ExecComp(['z=3.0*y'],
                                 y=numpy.zeros(size),
                                 z=numpy.zeros(size)))

        prob = Problem()
        prob.model = group
        prob.model.ln_solver = LinearBlockGS()
        prob.model.connect('P.x', 'C1.x')
        prob.model.connect('C1.y', 'C2.y')

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['P.x'], ['C2.z'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['C2.z']['P.x'], numpy.eye(size)*6.0, 1e-6)

        J = prob.calc_gradient(['P.x'], ['C2.z'], mode='rev', return_format='dict')
        assert_rel_error(self, J['C2.z']['P.x'], numpy.eye(size)*6.0, 1e-6)

    def test_fan_out_grouped(self):
        size = 3
        prob = Problem()
        prob.model = root = Group()
        root.add('P', IndepVarComp('x', numpy.ones(size, dtype=float)))
        root.add('C1', DistribExecComp(['y=3.0*x'], arr_size=size,
                                       x=numpy.zeros(size, dtype=float),
                                       y=numpy.zeros(size, dtype=float)))
        sub = root.add('sub', ParallelGroup())
        sub.add('C2', ExecComp('y=1.5*x',
                               x=numpy.zeros(size),
                               y=numpy.zeros(size)))
        sub.add('C3', ExecComp(['y=5.0*x'],
                               x=numpy.zeros(size, dtype=float),
                               y=numpy.zeros(size, dtype=float)))

        root.add('C2', ExecComp(['y=x'],
                                x=numpy.zeros(size, dtype=float),
                                y=numpy.zeros(size, dtype=float)))
        root.add('C3', ExecComp(['y=x'],
                                x=numpy.zeros(size, dtype=float),
                                y=numpy.zeros(size, dtype=float)))
        root.connect('sub.C2.y', 'C2.x')
        root.connect('sub.C3.y', 'C3.x')

        root.connect("C1.y", "sub.C2.x")
        root.connect("C1.y", "sub.C3.x")
        root.connect("P.x", "C1.x")

        root.ln_solver = LinearBlockGS()
        sub.ln_solver = LinearBlockGS()

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['P.x'], ['C2.y', "C3.y"], mode='fwd', return_format='dict')
        assert_rel_error(self, J['C2.y']['P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y']['P.x'], numpy.eye(size)*15.0, 1e-6)

        J = prob.calc_gradient(['P.x'], ['C2.y', "C3.y"], mode='rev', return_format='dict')
        assert_rel_error(self, J['C2.y']['P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y']['P.x'], numpy.eye(size)*15.0, 1e-6)

    def test_fan_in_grouped(self):
        size = 3

        prob = Problem()
        prob.model = root = Group()

        root.add('P1', IndepVarComp('x', numpy.ones(size, dtype=float)))
        root.add('P2', IndepVarComp('x', numpy.ones(size, dtype=float)))
        sub = root.add('sub', ParallelGroup())

        sub.add('C1', ExecComp(['y=-2.0*x'],
                               x=numpy.zeros(size, dtype=float),
                               y=numpy.zeros(size, dtype=float)))
        sub.add('C2', ExecComp(['y=5.0*x'],
                               x=numpy.zeros(size, dtype=float),
                               y=numpy.zeros(size, dtype=float)))
        root.add('C3', DistribExecComp(['y=3.0*x1+7.0*x2'], arr_size=size,
                                       x1=numpy.zeros(size, dtype=float),
                                       x2=numpy.zeros(size, dtype=float),
                                       y=numpy.zeros(size, dtype=float)))
        root.add('C4', ExecComp(['y=x'],
                                x=numpy.zeros(size, dtype=float),
                                y=numpy.zeros(size, dtype=float)))

        root.connect("sub.C1.y", "C3.x1")
        root.connect("sub.C2.y", "C3.x2")
        root.connect("P1.x", "sub.C1.x")
        root.connect("P2.x", "sub.C2.x")
        root.connect("C3.y", "C4.x")

        root.ln_solver = LinearBlockGS()
        sub.ln_solver = LinearBlockGS()

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['P1.x', 'P2.x'], ['C4.y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['C4.y']['P1.x'], numpy.eye(size)*-6.0, 1e-6)
        assert_rel_error(self, J['C4.y']['P2.x'], numpy.eye(size)*35.0, 1e-6)

        J = prob.calc_gradient(['P1.x', 'P2.x'], ['C4.y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['C4.y']['P1.x'], numpy.eye(size)*-6.0, 1e-6)
        assert_rel_error(self, J['C4.y']['P2.x'], numpy.eye(size)*35.0, 1e-6)

    def test_src_indices_error(self):
        size = 3
        group = Group()
        P = group.add('P', IndepVarComp('x', numpy.ones(size)))
        C1 = group.add('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                             x=numpy.zeros(size),
                                             y=numpy.zeros(size)))
        C2 = group.add('C2', ExecComp(['z=3.0*y'],
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
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "'C1.y' is a distributed variable"
                                       " and may not be used as a design var,"
                                       " objective, or constraint.")
        else:
            if MPI:
                self.fail("Exception expected")


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
