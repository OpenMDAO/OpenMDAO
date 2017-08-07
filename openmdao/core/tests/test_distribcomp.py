from __future__ import print_function

import unittest
import time

import six
import numpy as np

from openmdao.api import Problem, ExplicitComponent, Group, IndepVarComp
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


def take_nth(rank, size, seq):
    """Return an iterator over the sequence that returns every
    nth element of seq based on the given rank within a group of
    the given size.  For example, if size = 2, a rank of 0 returns
    even indexed elements and a rank of 1 returns odd indexed elements.
    """
    assert(rank < size)
    it = iter(seq)
    while True:
        for proc in range(size):
            if rank == proc:
                yield six.next(it)
            else:
                six.next(it)


class InOutArrayComp(ExplicitComponent):

    def __init__(self, arr_size=10):
        super(InOutArrayComp, self).__init__()
        self.delay = 0.01
        self.arr_size = arr_size

    def setup(self):
        self.add_input('invec', np.ones(self.arr_size, float))
        self.add_output('outvec', np.ones(self.arr_size, float))

    def compute(self, inputs, outputs):
        time.sleep(self.delay)
        outputs['outvec'] = inputs['invec'] * 2.


class DistribCompSimple(ExplicitComponent):
    """Uses 2 procs but takes full input vars"""

    def __init__(self, arr_size=10):
        super(DistribCompSimple, self).__init__()

        self.arr_size = arr_size

    def setup(self):
        self.add_input('invec', np.ones(self.arr_size, float))
        self.add_output('outvec', np.ones(self.arr_size, float))

    def compute(self, inputs, outputs):
        if MPI and self.comm != MPI.COMM_NULL:
            if rank == 0:
                outvec = inputs['invec'] * 0.25
            elif rank == 1:
                outvec = inputs['invec'] * 0.5

            # now combine vecs from different processes
            both = np.zeros((2, len(outvec)))
            self.comm.Allgather(outvec, both)

            # add both together to get our output
            outputs['outvec'] = both[0, :] + both[1, :]
        else:
            outputs['outvec'] = inputs['invec'] * 0.75

    def get_req_procs(self):
        return (2, 2)


class DistribInputComp(ExplicitComponent):
    """Uses 2 procs and takes input var slices"""
    def __init__(self, arr_size=11):
        super(DistribInputComp, self).__init__()
        self.arr_size = arr_size
        self.distributed = True

    def compute(self, inputs, outputs):
        if MPI:
            self.comm.Allgatherv(inputs['invec']*2.0,
                                 [outputs['outvec'], self.sizes,
                                  self.offsets, MPI.DOUBLE])
        else:
            outputs['outvec'] = inputs['invec'] * 2.0

    def setup(self):
        comm = self.comm
        rank = comm.rank

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = self.offsets[rank]
        end = start + self.sizes[rank]

        self.add_input('invec', np.ones(self.sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(self.arr_size, float))

    def get_req_procs(self):
        return (2, 2)


class DistribOverlappingInputComp(ExplicitComponent):
    """Uses 2 procs and takes input var slices"""
    def __init__(self, arr_size=11):
        super(DistribOverlappingInputComp, self).__init__()
        self.arr_size = arr_size
        self.distributed = True

    def compute(self, inputs, outputs):
        outputs['outvec'][:] = 0
        if MPI:
            outs = self.comm.allgather(inputs['invec'] * 2.0)
            outputs['outvec'][:8] = outs[0]
            outputs['outvec'][4:11] += outs[1]
        else:
            outs = inputs['invec'] * 2.0
            outputs['outvec'][:8] = outs[:8]
            outputs['outvec'][4:11] += outs[4:11]

    def setup(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs"""

        comm = self.comm
        rank = comm.rank

        #need to initialize the input to have the correct local size
        if rank == 0:
            size = 8
            start = 0
            end = 8
        else:
            size = 7
            start = 4
            end = 11

        self.add_output('outvec', np.zeros(self.arr_size, float))
        self.add_input('invec', np.ones(size, float),
                       src_indices=np.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class DistribInputDistribOutputComp(ExplicitComponent):
    """Uses 2 procs and takes input var slices."""
    def __init__(self, arr_size=11):
        super(DistribInputDistribOutputComp, self).__init__()
        self.arr_size = arr_size
        self.distributed = True

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']*2.0

    def setup(self):

        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        self.add_input('invec', np.ones(sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(sizes[rank], float))

    def get_req_procs(self):
        return (2, 2)


class DistribNoncontiguousComp(ExplicitComponent):
    """Uses 2 procs and takes non-contiguous input var slices and has output
    var slices as well
    """
    def __init__(self, arr_size=11):
        super(DistribNoncontiguousComp, self).__init__()
        self.arr_size = arr_size
        self.distributed = True

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']*2.0

    def setup(self):

        comm = self.comm
        rank = comm.rank

        idxs = list(take_nth(rank, comm.size, range(self.arr_size)))

        self.add_input('invec', np.ones(len(idxs), float),
                       src_indices=idxs)
        self.add_output('outvec', np.ones(len(idxs), float))

    def get_req_procs(self):
        return 2, 2


class DistribGatherComp(ExplicitComponent):
    """Uses 2 procs gathers a distrib input into a full output"""

    def __init__(self, arr_size=11):
        super(DistribGatherComp, self).__init__()
        self.arr_size = arr_size
        self.distributed = True

    def compute(self, inputs, outputs):
        if MPI:
            self.comm.Allgatherv(inputs['invec'],
                                 [outputs['outvec'], self.sizes,
                                     self.offsets, MPI.DOUBLE])
        else:
            outputs['outvec'] = inputs['invec']

    def setup(self):

        comm = self.comm
        rank = comm.rank

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size,
                                                       self.arr_size)
        start = self.offsets[rank]
        end = start + self.sizes[rank]

        #need to initialize the variable to have the correct local size
        self.add_input('invec', np.ones(self.sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(self.arr_size, float))

    def get_req_procs(self):
        return 2, 2


class NonDistribGatherComp(ExplicitComponent):
    """Uses 2 procs gathers a distrib output into a full input"""
    def __init__(self, size):
        super(NonDistribGatherComp, self).__init__()
        self.size = size

    def setup(self):
        self.add_input('invec', np.ones(self.size, float))
        self.add_output('outvec', np.ones(self.size, float))

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITests(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_full_in_out(self):
        size = 11

        p = Problem(model=Group())
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(size))
        C2 = top.add_subsystem("C2", DistribCompSimple(size))
        top.connect('C1.outvec', 'C2.invec')

        p.setup(vector_class=PETScVector, check=False)

        C1._inputs['invec'] = np.ones(size, float) * 5.0

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.ones(size, float)*7.5))

    def test_distrib_idx_in_full_out(self):
        size = 11

        p = Problem(model=Group())
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(size))
        C2 = top.add_subsystem("C2", DistribInputComp(size))
        top.connect('C1.outvec', 'C2.invec')
        p.setup(vector_class=PETScVector, check=False)

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_distrib_idx_in_distrb_idx_out(self):
        # normal comp to distrib comp to distrb gather comp
        size = 3

        p = Problem(model=Group())
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(size))
        C2 = top.add_subsystem("C2", DistribInputDistribOutputComp(size))
        C3 = top.add_subsystem("C3", DistribGatherComp(size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(vector_class=PETScVector, check=False)

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C3._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_distribcomp_feature(self):
        from openmdao.utils.array_utils import evenly_distrib_idxs

        size = 15

        class DistribComp(ExplicitComponent):
            def __init__(self, size):
                super(DistribComp, self).__init__()
                self.size = size
                self.distributed = True

            def compute(self, inputs, outputs):
                if self.comm.rank == 0:
                    outputs['outvec'] = inputs['invec'] * 2.0
                else:
                    outputs['outvec'] = inputs['invec'] * -3.0

            def setup(self):
                comm = self.comm
                rank = comm.rank

                # this results in 8 entries for proc 0 and 7 entries for proc 1 when using 2 processes.
                sizes, offsets = evenly_distrib_idxs(comm.size, self.size)
                start = offsets[rank]
                end = start + sizes[rank]

                self.add_input('invec', np.ones(sizes[rank], float),
                               src_indices=np.arange(start, end, dtype=int))
                self.add_output('outvec', np.ones(sizes[rank], float))

            def get_req_procs(self):
                # require min of 2 processes, max of 5
                return 2, 5

        class Summer(ExplicitComponent):
            """Sums a distributed input."""

            def __init__(self, size):
                super(Summer, self).__init__()
                self.size = size

            def setup(self):
                # this results in 8 entries for proc 0 and 7 entries for proc 1
                # when using 2 processes.
                sizes, offsets = evenly_distrib_idxs(self.comm.size, self.size)
                start = offsets[rank]
                end = start + sizes[rank]

                # NOTE: you must specify src_indices here for the input. Otherwise,
                #       you'll connect the input to [0:local_input_size] of the
                #       full distributed output!
                self.add_input('invec', np.ones(sizes[self.comm.rank], float),
                               src_indices=np.arange(start, end, dtype=int))
                self.add_output('out', 0.0)

            def compute(self, inputs, outputs):
                data = np.zeros(1)
                data[0] = np.sum(self._inputs['invec'])
                total = np.zeros(1)
                self.comm.Allreduce(data, total, op=MPI.SUM)
                self._outputs['out'] = total[0]

        p = Problem(model=Group())
        top = p.model
        top.add_subsystem("indep", IndepVarComp('x', np.zeros(size)))
        top.add_subsystem("C2", DistribComp(size))
        top.add_subsystem("C3", Summer(size))

        top.connect('indep.x', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')

        p.setup(vector_class=PETScVector)

        p['indep.x'] = np.ones(size)

        p.run_model()

        assert_rel_error(self, p['C3.out'], -5.)

    def test_noncontiguous_idxs(self):
        # take even input indices in 0 rank and odd ones in 1 rank
        size = 11

        p = Problem(model=Group())
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(size))
        C2 = top.add_subsystem("C2", DistribNoncontiguousComp(size))
        C3 = top.add_subsystem("C3", DistribGatherComp(size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(vector_class=PETScVector, check=False)

        C1._inputs['invec'] = np.array(range(size), float)

        p.run_model()

        if MPI:
            if self.comm.rank == 0:
                self.assertTrue(all(C2._outputs['outvec'] == np.array(list(take_nth(0, 2, range(size))), 'f')*4))
            else:
                self.assertTrue(all(C2._outputs['outvec'] == np.array(list(take_nth(1, 2, range(size))), 'f')*4))

            full_list = list(take_nth(0, 2, range(size))) + list(take_nth(1, 2, range(size)))
            self.assertTrue(all(C3._outputs['outvec'] == np.array(full_list, 'f')*4))
        else:
            self.assertTrue(all(C2._outputs['outvec'] == C1._outputs['outvec']*2.))
            self.assertTrue(all(C3._outputs['outvec'] == C2._outputs['outvec']))

    def test_overlapping_inputs_idxs(self):
        # distrib comp with src_indices that overlap, i.e. the same
        # entries are distributed to multiple processes
        size = 11

        p = Problem(model=Group())
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(size))
        C2 = top.add_subsystem("C2", DistribOverlappingInputComp(size))
        top.connect('C1.outvec', 'C2.invec')
        p.setup(vector_class=PETScVector, check=False)

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'][:4] == np.array(range(size, 0, -1), float)[:4]*4))
        self.assertTrue(all(C2._outputs['outvec'][8:] == np.array(range(size, 0, -1), float)[8:]*4))

        # overlapping part should be double size of the rest
        self.assertTrue(all(C2._outputs['outvec'][4:8] == np.array(range(size, 0, -1), float)[4:8]*8))

    def test_nondistrib_gather(self):
        # regular comp --> distrib comp --> regular comp.  last comp should
        # automagically gather the full vector without declaring src_indices
        size = 11

        p = Problem(model=Group())
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(size))
        C2 = top.add_subsystem("C2", DistribInputDistribOutputComp(size))
        C3 = top.add_subsystem("C3", NonDistribGatherComp(size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(vector_class=PETScVector, check=False)

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        if self.comm.rank == 0:
            self.assertTrue(all(C3._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestGroupMPI(unittest.TestCase):
    N_PROCS = 2

    def test_promote_distrib(self):

        class MyComp(ExplicitComponent):
            def setup(self):
                # decide what parts of the array we want based on our rank
                if self.comm.rank == 0:
                    idxs = [0, 1, 2]
                else:
                    # use [3, -1] here rather than [3, 4] just to show that we
                    # can use negative indices.
                    idxs = [3, -1]

                self.add_input('x', np.ones(len(idxs)), src_indices=idxs)
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        p = Problem(model=Group())

        #import wingdbstub

        p.model.add_subsystem('indep', IndepVarComp('x', np.arange(5, dtype=float)),
                              promotes_outputs=['x'])
        C1 = p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.set_solver_print(level=0)
        p.setup(PETScVector)
        p.run_model()
        if C1.comm.rank == 0:
            assert_rel_error(self, p['C1.x'], np.arange(3, dtype=float))
            assert_rel_error(self, p['C1.y'], 6.)
        else:
            assert_rel_error(self, p['C1.x'], np.arange(3, 5, dtype=float))
            assert_rel_error(self, p['C1.y'], 14.)


if __name__ == '__main__':
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
