from __future__ import print_function

import time

import numpy as np

from openmdao.api import Problem, Component, Group
from openmdao.core.mpi_wrap import MPI
from openmdao.util.array_util import evenly_distrib_idxs
from openmdao.test.mpi_util import MPITestCase
import six

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    rank = MPI.COMM_WORLD.rank
else:
    from openmdao.core.basic_impl import BasicImpl as impl
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


class InOutArrayComp(Component):

    def __init__(self, arr_size=10):
        super(InOutArrayComp, self).__init__()
        self.delay = 0.01

        self.add_param('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        time.sleep(self.delay)
        unknowns['outvec'] = params['invec'] * 2.


class DistribCompSimple(Component):
    """Uses 2 procs but takes full input vars"""

    def __init__(self, arr_size=10):
        super(DistribCompSimple, self).__init__()

        self.add_param('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        if MPI and self.comm != MPI.COMM_NULL:
            if rank == 0:
                outvec = params['invec'] * 0.25
            elif rank == 1:
                outvec = params['invec'] * 0.5

            # now combine vecs from different processes
            both = np.zeros((2, len(outvec)))
            self.comm.Allgather(outvec, both)

            # add both together to get our output
            unknowns['outvec'] = both[0, :] + both[1, :]
        else:
            unknowns['outvec'] = params['invec'] * 0.75

    def get_req_procs(self):
        return (2, 2)


class DistribInputComp(Component):
    """Uses 2 procs and takes input var slices"""
    def __init__(self, arr_size=11):
        super(DistribInputComp, self).__init__()
        self.arr_size = arr_size
        self.add_param('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        if MPI:
            self.comm.Allgatherv(params['invec']*2.0,
                                 [unknowns['outvec'], self.sizes,
                                  self.offsets, MPI.DOUBLE])
        else:
            unknowns['outvec'] = params['invec'] * 2.0

    def setup_distrib(self):
        # this is called at the beginning of _setup_variables, so we can
        # add new params/unknowns here.
        comm = self.comm
        rank = comm.rank

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = self.offsets[rank]
        end = start + self.sizes[rank]

        #need to initialize the param to have the correct local size
        self.set_var_indices('invec', val=np.ones(self.sizes[rank], float),
                             src_indices=np.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class DistribOverlappingInputComp(Component):
    """Uses 2 procs and takes input var slices"""
    def __init__(self, arr_size=11):
        super(DistribOverlappingInputComp, self).__init__()
        self.arr_size = arr_size
        self.add_param('invec', np.zeros(arr_size, float))
        self.add_output('outvec', np.zeros(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['outvec'][:] = 0
        if MPI:
            outs = self.comm.allgather(params['invec'] * 2.0)
            unknowns['outvec'][:8] = outs[0]
            unknowns['outvec'][4:11] += outs[1]
        else:
            outs = params['invec'] * 2.0
            unknowns['outvec'][:8] = outs[:8]
            unknowns['outvec'][4:11] += outs[4:11]

    def setup_distrib(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs"""

        comm = self.comm
        rank = comm.rank

        #need to initialize the param to have the correct local size
        if rank == 0:
            size = 8
            start = 0
            end = 8
        else:
            size = 7
            start = 4
            end = 11

        self.set_var_indices('invec', val=np.ones(size, float),
                             src_indices=np.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class DistribInputDistribOutputComp(Component):
    """Uses 2 procs and takes input var slices and has output var slices as well"""
    def __init__(self, arr_size=11):
        super(DistribInputDistribOutputComp, self).__init__()
        self.arr_size = arr_size
        self.add_param('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['outvec'] = params['invec']*2.0

    def setup_distrib(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs. Returns a dict of
        index arrays keyed to variable names.
        """

        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        self.set_var_indices('invec', val=np.ones(sizes[rank], float),
                             src_indices=np.arange(start, end, dtype=int))
        self.set_var_indices('outvec', val=np.ones(sizes[rank], float),
                             src_indices=np.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class DistribNoncontiguousComp(Component):
    """Uses 2 procs and takes non-contiguous input var slices and has output
    var slices as well
    """
    def __init__(self, arr_size=11):
        super(DistribNoncontiguousComp, self).__init__()
        self.arr_size = arr_size
        self.add_param('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['outvec'] = params['invec']*2.0

    def setup_distrib(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs. Returns a dict of
        index arrays keyed to variable names.
        """

        comm = self.comm
        rank = comm.rank

        idxs = list(take_nth(rank, comm.size, range(self.arr_size)))

        self.set_var_indices('invec', val=np.ones(len(idxs), float),
                             src_indices=idxs)
        self.set_var_indices('outvec', val=np.ones(len(idxs), float),
                             src_indices=idxs)

    def get_req_procs(self):
        return (2, 2)


class DistribGatherComp(Component):
    """Uses 2 procs gathers a distrib input into a full output"""

    def __init__(self, arr_size=11):
        super(DistribGatherComp, self).__init__()
        self.arr_size = arr_size
        self.add_param('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        if MPI:
            self.comm.Allgatherv(params['invec'],
                                 [unknowns['outvec'], self.sizes,
                                     self.offsets, MPI.DOUBLE])
        else:
            unknowns['outvec'] = params['invec']

    def setup_distrib(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs. Returns a dict of
        index arrays keyed to variable names.
        """
        comm = self.comm
        rank = comm.rank

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size,
                                                       self.arr_size)
        start = self.offsets[rank]
        end = start + self.sizes[rank]

        #need to initialize the variable to have the correct local size
        self.set_var_indices('invec', val=np.ones(self.sizes[rank], float),
                             src_indices=np.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class NonDistribGatherComp(Component):
    """Uses 2 procs gathers a distrib input into a full output"""
    def __init__(self, size):
        super(NonDistribGatherComp, self).__init__()
        self.add_param('invec', np.ones(size, float))
        self.add_output('outvec', np.ones(size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['outvec'] = params['invec']


class MPITests(MPITestCase):

    N_PROCS = 2

    def test_distrib_full_in_out(self):
        size = 11

        p = Problem(root=Group(), impl=impl)
        top = p.root
        top.add("C1", InOutArrayComp(size))
        top.add("C2", DistribCompSimple(size))
        top.connect('C1.outvec', 'C2.invec')

        p.setup(check=False)

        top.C1.params['invec'] = np.ones(size, float) * 5.0

        p.run()

        self.assertTrue(all(top.C2.unknowns['outvec'] == np.ones(size, float)*7.5))

    def test_distrib_idx_in_full_out(self):
        size = 11

        p = Problem(root=Group(), impl=impl)
        top = p.root
        top.add("C1", InOutArrayComp(size))
        top.add("C2", DistribInputComp(size))
        top.connect('C1.outvec', 'C2.invec')
        p.setup(check=False)

        top.C1.params['invec'] = np.array(range(size, 0, -1), float)

        p.run()

        self.assertTrue(all(top.C2.unknowns['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_distrib_idx_in_distrb_idx_out(self):
        # normal comp to distrib comp to distrb gather comp
        size = 3

        p = Problem(root=Group(), impl=impl)
        top = p.root
        top.add("C1", InOutArrayComp(size))
        top.add("C2", DistribInputDistribOutputComp(size))
        top.add("C3", DistribGatherComp(size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(check=False)

        top.C1.params['invec'] = np.array(range(size, 0, -1), float)

        p.run()

        self.assertTrue(all(top.C3.unknowns['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_noncontiguous_idxs(self):
        # take even input indices in 0 rank and odd ones in 1 rank
        size = 11

        p = Problem(root=Group(), impl=impl)
        top = p.root
        top.add("C1", InOutArrayComp(size))
        top.add("C2", DistribNoncontiguousComp(size))
        top.add("C3", DistribGatherComp(size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(check=False)

        top.C1.params['invec'] = np.array(range(size), float)

        p.run()

        if MPI:
            if self.comm.rank == 0:
                self.assertTrue(all(top.C2.unknowns['outvec'] == np.array(list(take_nth(0, 2, range(size))), 'f')*4))
            else:
                self.assertTrue(all(top.C2.unknowns['outvec'] == np.array(list(take_nth(1, 2, range(size))), 'f')*4))

            full_list = list(take_nth(0, 2, range(size))) + list(take_nth(1, 2, range(size)))
            self.assertTrue(all(top.C3.unknowns['outvec'] == np.array(full_list, 'f')*4))
        else:
            self.assertTrue(all(top.C2.unknowns['outvec'] == top.C1.unknowns['outvec']*2.))
            self.assertTrue(all(top.C3.unknowns['outvec'] == top.C2.unknowns['outvec']))

    def test_overlapping_inputs_idxs(self):
        # distrib comp with src_indices that overlap, i.e. the same
        # entries are distributed to multiple processes
        size = 11

        p = Problem(root=Group(), impl=impl)
        top = p.root
        top.add("C1", InOutArrayComp(size))
        top.add("C2", DistribOverlappingInputComp(size))
        top.connect('C1.outvec', 'C2.invec')
        p.setup(check=False)

        top.C1.params['invec'] = np.array(range(size, 0, -1), float)

        p.run()

        self.assertTrue(all(top.C2.unknowns['outvec'][:4] == np.array(range(size, 0, -1), float)[:4]*4))
        self.assertTrue(all(top.C2.unknowns['outvec'][8:] == np.array(range(size, 0, -1), float)[8:]*4))

        # overlapping part should be double size of the rest
        self.assertTrue(all(top.C2.unknowns['outvec'][4:8] == np.array(range(size, 0, -1), float)[4:8]*8))

    def test_nondistrib_gather(self):
        # regular comp --> distrib comp --> regular comp.  last comp should
        # automagically gather the full vector without declaring src_indices
        size = 11

        p = Problem(root=Group(), impl=impl)
        top = p.root
        top.add("C1", InOutArrayComp(size))
        top.add("C2", DistribInputDistribOutputComp(size))
        top.add("C3", NonDistribGatherComp(size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(check=False)

        top.C1.params['invec'] = np.array(range(size, 0, -1), float)

        p.run()

        if rank == 0:
            self.assertTrue(all(top.C3.unknowns['outvec'] == np.array(range(size, 0, -1), float)*4))


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
