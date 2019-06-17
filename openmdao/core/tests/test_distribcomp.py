from __future__ import print_function

import unittest
import time

import numpy as np

from openmdao.api import Problem, ExplicitComponent, Group, ExecComp, ParallelGroup
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs, take_nth
from openmdao.utils.assert_utils import assert_rel_error, assert_warning

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

if MPI:
    rank = MPI.COMM_WORLD.rank
    commsize = MPI.COMM_WORLD.size
else:
    rank = 0
    commsize = 1


class InOutArrayComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('arr_size', types=int, default=10,
                             desc="Size of input and output vectors.")

        self.options.declare('delay', types=float, default=.01,
                             desc="Time to sleep in compute function.")

    def setup(self):
        arr_size = self.options['arr_size']

        self.add_input('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def compute(self, inputs, outputs):
        time.sleep(self.options['delay'])
        outputs['outvec'] = inputs['invec'] * 2.


class DistribCompSimple(ExplicitComponent):
    """Uses 2 procs but takes full input vars"""

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=10,
                             desc="Size of input and output vectors.")

    def setup(self):
        arr_size = self.options['arr_size']

        self.add_input('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

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


class DistribInputComp(ExplicitComponent):
    """Uses 2 procs and takes input var slices"""

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

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

        arr_size = self.options['arr_size']

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size, arr_size)
        start = self.offsets[rank]
        end = start + self.sizes[rank]

        self.add_input('invec', np.ones(self.sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(arr_size, float), shape=np.int32(arr_size))


class DistribOverlappingInputComp(ExplicitComponent):
    """Uses 2 procs and takes input var slices"""

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

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

        arr_size = self.options['arr_size']

        # need to initialize the input to have the correct local size
        if rank == 0:
            size = 8
            start = 0
            end = 8
        else:
            size = 7
            start = 4
            end = 11

        self.add_output('outvec', np.zeros(arr_size, float))
        self.add_input('invec', np.ones(size, float),
                       src_indices=np.arange(start, end, dtype=int))


class DistribInputDistribOutputComp(ExplicitComponent):
    """Uses 2 procs and takes input var slices."""

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']*2.0

    def setup(self):

        comm = self.comm
        rank = comm.rank

        arr_size = self.options['arr_size']

        sizes, offsets = evenly_distrib_idxs(comm.size, arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        self.add_input('invec', np.ones(sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(sizes[rank], float))


class DistribNoncontiguousComp(ExplicitComponent):
    """Uses 2 procs and takes non-contiguous input var slices and has output
    var slices as well
    """

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']*2.0

    def setup(self):

        comm = self.comm
        rank = comm.rank

        arr_size = self.options['arr_size']

        idxs = list(take_nth(rank, comm.size, range(arr_size)))

        self.add_input('invec', np.ones(len(idxs), float),
                       src_indices=idxs)
        self.add_output('outvec', np.ones(len(idxs), float))


class DistribGatherComp(ExplicitComponent):
    """Uses 2 procs gathers a distrib input into a full output"""

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

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

        arr_size = self.options['arr_size']

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size,
                                                       arr_size)
        start = self.offsets[rank]
        end = start + self.sizes[rank]

        # need to initialize the variable to have the correct local size
        self.add_input('invec', np.ones(self.sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(arr_size, float))


class NonDistribGatherComp(ExplicitComponent):
    """Uses 2 procs gathers a distrib output into a full input"""

    def initialize(self):
        self.options.declare('size', types=int, default=1,
                             desc="Size of input and output vectors.")

    def setup(self):
        size = self.options['size']

        self.add_input('invec', np.ones(size, float))
        self.add_output('outvec', np.ones(size, float))

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']


@unittest.skipIf(PETScVector is not None, "Only runs when PETSc is not available")
class NOMPITests(unittest.TestCase):

    def test_distrib_idx_in_full_out(self):
        size = 11

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')

        msg = "The 'distributed' option is set to True for Component C2, " \
              "but there is no distributed vector implementation (MPI/PETSc) " \
              "available. The default non-distributed vectors will be used."

        with assert_warning(UserWarning, msg):
            p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITests(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_full_in_out(self):
        size = 11

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribCompSimple(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')

        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.ones(size, float) * 5.0

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.ones(size, float)*7.5))

    def test_distrib_idx_in_full_out(self):
        size = 11

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')

        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_distrib_1D_dist_output(self):
        size = 11

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputComp(arr_size=size))
        C3 = top.add_subsystem("C3", ExecComp("y=x", x=np.zeros(size*commsize),
                                              y=np.zeros(size*commsize)))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.x')
        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_distrib_idx_in_distrb_idx_out(self):
        # normal comp to distrib comp to distrb gather comp
        size = 3

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputDistribOutputComp(arr_size=size))
        C3 = top.add_subsystem("C3", DistribGatherComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C3._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_noncontiguous_idxs(self):
        # take even input indices in 0 rank and odd ones in 1 rank
        size = 11

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribNoncontiguousComp(arr_size=size))
        C3 = top.add_subsystem("C3", DistribGatherComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.array(range(size), float)

        p.run_model()

        if MPI:
            if self.comm.rank == 0:
                self.assertTrue(all(C2._outputs['outvec'] ==
                                    np.array(list(take_nth(0, 2, range(size))), 'f')*4))
            else:
                self.assertTrue(all(C2._outputs['outvec'] ==
                                    np.array(list(take_nth(1, 2, range(size))), 'f')*4))

            full_list = list(take_nth(0, 2, range(size))) + list(take_nth(1, 2, range(size)))
            self.assertTrue(all(C3._outputs['outvec'] == np.array(full_list, 'f')*4))
        else:
            self.assertTrue(all(C2._outputs['outvec'] == C1._outputs['outvec']*2.))
            self.assertTrue(all(C3._outputs['outvec'] == C2._outputs['outvec']))

    @unittest.skipUnless(MPI, "MPI is not active.")
    def test_overlapping_inputs_idxs(self):
        # distrib comp with src_indices that overlap, i.e. the same
        # entries are distributed to multiple processes
        size = 11

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribOverlappingInputComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')
        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'][:4] == np.array(range(size, 0, -1), float)[:4]*4))
        self.assertTrue(all(C2._outputs['outvec'][8:] == np.array(range(size, 0, -1), float)[8:]*4))

        # overlapping part should be double size of the rest
        self.assertTrue(all(C2._outputs['outvec'][4:8] ==
                            np.array(range(size, 0, -1), float)[4:8]*8))

    def test_nondistrib_gather(self):
        # regular comp --> distrib comp --> regular comp.  last comp should
        # automagically gather the full vector without declaring src_indices
        size = 11

        p = Problem()
        top = p.model
        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputDistribOutputComp(arr_size=size))
        C3 = top.add_subsystem("C3", NonDistribGatherComp(size=size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        if MPI and self.comm.rank == 0:
            self.assertTrue(all(C3._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))


class DeprecatedMPITests(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_idx_in_full_out_deprecated(self):

        class DeprecatedDistribInputComp(ExplicitComponent):
            """Deprecated version of DistribInputComp, uses attribute instead of option."""

            def __init__(self, arr_size=11):
                super(DeprecatedDistribInputComp, self).__init__()
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
                self.add_output('outvec', np.ones(self.arr_size, float),
                                shape=np.int32(self.arr_size))

        size = 11

        p = Problem()
        top = p.model

        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))

        # check deprecation on setter & getter
        msg = "The 'distributed' property provides backwards compatibility " \
              "with OpenMDAO <= 2.4.0 ; use the 'distributed' option instead."

        with assert_warning(DeprecationWarning, msg):
            C2 = top.add_subsystem("C2", DeprecatedDistribInputComp(arr_size=size))

        with assert_warning(DeprecationWarning, msg):
            C2.distributed

        # continue to make sure everything still works with the deprecation
        top.connect('C1.outvec', 'C2.invec')

        # Conclude setup but don't run model.
        msg = "The 'distributed' option is set to True for Component C2, " \
              "but there is no distributed vector implementation (MPI/PETSc) " \
              "available. The default non-distributed vectors will be used."

        if PETScVector is None:
            with assert_warning(UserWarning, msg):
                p.setup(check=False)
        else:
            p.setup(check=False)

        p.final_setup()

        C1._inputs['invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))


@unittest.skipUnless(PETScVector, "PETSc is required.")
@unittest.skipUnless(MPI, "MPI is required.")
class ProbRemoteTests(unittest.TestCase):

    N_PROCS = 4

    def test_prob_getitem_err(self):
        size = 3

        p = Problem()
        top = p.model
        par = top.add_subsystem('par', ParallelGroup())
        C1 = par.add_subsystem("C1", DistribInputDistribOutputComp(arr_size=size))
        C2 = par.add_subsystem("C2", DistribInputDistribOutputComp(arr_size=size))
        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        if C1 in p.model.par._subsystems_myproc:
            C1._inputs['invec'] = np.array(range(C1._inputs._data.size, 0, -1), float)

        if C2 in p.model.par._subsystems_myproc:
            C2._inputs['invec'] = np.array(range(C2._inputs._data.size, 0, -1), float) * 3

        p.run_model()

        # test that getitem from Problem on a distrib var raises an exception
        with self.assertRaises(Exception) as context:
            ans = p.get_val('par.C2.invec', get_remote=True)
        self.assertEqual(str(context.exception),
                         "Retrieval of the full distributed variable 'par.C2.invec' is not supported.")
        with self.assertRaises(Exception) as context:
            ans = p.get_val('par.C2.outvec', get_remote=True)
        self.assertEqual(str(context.exception),
                         "Retrieval of the full distributed variable 'par.C2.outvec' is not supported.")
        with self.assertRaises(Exception) as context:
            ans = p.get_val('par.C1.invec', get_remote=True)
        self.assertEqual(str(context.exception),
                         "Retrieval of the full distributed variable 'par.C1.invec' is not supported.")
        with self.assertRaises(Exception) as context:
            ans = p.get_val('par.C1.outvec', get_remote=True)
        self.assertEqual(str(context.exception),
                         "Retrieval of the full distributed variable 'par.C1.outvec' is not supported.")


@unittest.skipUnless(PETScVector, "PETSc is required.")
@unittest.skipUnless(MPI, "MPI is required.")
class MPIFeatureTests(unittest.TestCase):

    N_PROCS = 2

    def test_distribcomp_feature(self):
        import numpy as np

        from openmdao.api import Problem, ExplicitComponent, Group, IndepVarComp
        from openmdao.utils.mpi import MPI
        from openmdao.utils.array_utils import evenly_distrib_idxs

        if not MPI:
            raise unittest.SkipTest()

        rank = MPI.COMM_WORLD.rank
        size = 15

        class DistribComp(ExplicitComponent):
            def initialize(self):
                self.options['distributed'] = True

                self.options.declare('size', types=int, default=1,
                                     desc="Size of input and output vectors.")

            def setup(self):
                comm = self.comm
                rank = comm.rank

                size = self.options['size']

                # results in 8 entries for proc 0 and 7 entries for proc 1 when using 2 processes.
                sizes, offsets = evenly_distrib_idxs(comm.size, size)
                start = offsets[rank]
                end = start + sizes[rank]

                self.add_input('invec', np.ones(sizes[rank], float),
                               src_indices=np.arange(start, end, dtype=int))

                self.add_output('outvec', np.ones(sizes[rank], float))

            def compute(self, inputs, outputs):
                if self.comm.rank == 0:
                    outputs['outvec'] = inputs['invec'] * 2.0
                else:
                    outputs['outvec'] = inputs['invec'] * -3.0

        class Summer(ExplicitComponent):
            """Sums a distributed input."""

            def initialize(self):
                self.options.declare('size', types=int, default=1,
                                     desc="Size of input and output vectors.")

            def setup(self):
                comm = self.comm
                rank = comm.rank

                size = self.options['size']

                # this results in 8 entries for proc 0 and 7 entries for proc 1
                # when using 2 processes.
                sizes, offsets = evenly_distrib_idxs(comm.size, size)
                start = offsets[rank]
                end = start + sizes[rank]

                # NOTE: you must specify src_indices here for the input. Otherwise,
                #       you'll connect the input to [0:local_input_size] of the
                #       full distributed output!
                self.add_input('invec', np.ones(sizes[rank], float),
                               src_indices=np.arange(start, end, dtype=int))

                self.add_output('out', 0.0)

            def compute(self, inputs, outputs):
                data = np.zeros(1)
                data[0] = np.sum(inputs['invec'])

                total = np.zeros(1)
                self.comm.Allreduce(data, total, op=MPI.SUM)

                outputs['out'] = total[0]

        p = Problem()
        top = p.model
        top.add_subsystem("indep", IndepVarComp('x', np.zeros(size)))
        top.add_subsystem("C2", DistribComp(size=size))
        top.add_subsystem("C3", Summer(size=size))

        top.connect('indep.x', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')

        p.setup()

        p['indep.x'] = np.ones(size)

        p.run_model()

        assert_rel_error(self, p['C3.out'], -5.)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestGroupMPI(unittest.TestCase):
    N_PROCS = 2

    def test_promote_distrib(self):
        import numpy as np

        from openmdao.api import Problem, ExplicitComponent, IndepVarComp

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

        p = Problem()

        p.model.add_subsystem('indep', IndepVarComp('x', np.arange(5, dtype=float)),
                              promotes_outputs=['x'])

        p.model.add_subsystem('C1', MyComp(),
                              promotes_inputs=['x'])

        p.setup()
        p.run_model()

        # each rank holds the assigned portion of the input array
        assert_rel_error(self, p['C1.x'],
                         np.arange(3, dtype=float) if p.model.C1.comm.rank == 0 else
                         np.arange(3, 5, dtype=float))

        # the output in each rank is based on the local inputs
        assert_rel_error(self, p['C1.y'], 6. if p.model.C1.comm.rank == 0 else 14.)


if __name__ == '__main__':
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
