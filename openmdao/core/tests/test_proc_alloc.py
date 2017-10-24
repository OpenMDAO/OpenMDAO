from __future__ import print_function

import unittest

from openmdao.utils.mpi import MPI
from openmdao.core.default_allocator import DefaultAllocator


@unittest.skipIf(not MPI)
class ProcTestCase1(unittest.TestCase):

    N_PROCS = 1

    def test_proc(self):
        allocator = DefaultAllocator(parallel=True)

        proc_info = [
            (1, None, 4.5),
            (1, None, 1.0),
            (1, None, 2.0),
            (1, None, 4.0),
        ]

        weights = [w for _, _, w in proc_info]

        # normalize so that smallest weight is 1
        norm = weights / np.sum(weights)

        comm = MPI.COMM_WORLD

        if comm.rank == 0:
            print("size:", comm.size)
        try:
            isubs, sub_comm, sub_proc_range = allocator._divide_procs(proc_info, MPI.COMM_WORLD)
        except Exception as err:
            traceback.print_exc()
        print("  %d: %s %s" % (comm.rank, isubs, sub_proc_range))
