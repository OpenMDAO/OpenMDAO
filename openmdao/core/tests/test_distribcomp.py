
import unittest
import time

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.distributed_components import DistribComp, Summer
from openmdao.utils.mpi import MPI, multi_proc_exception_check
from openmdao.utils.array_utils import evenly_distrib_idxs, take_nth
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_partials
from openmdao.utils.om_warnings import DistributedComponentWarning
from openmdao.utils.variable_table import NA

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class InOutArrayComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('arr_size', types=int, default=10,
                             desc="Size of input and output vectors.")

        self.options.declare('delay', types=float, default=.01,
                             desc="Time to sleep in compute function.")

    def setup(self):
        arr_size = self.options['arr_size']

        self.add_input('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))
        self.declare_partials('outvec', 'invec', rows=np.arange(arr_size), cols=np.arange(arr_size))
    def compute(self, inputs, outputs):
        time.sleep(self.options['delay'])
        outputs['outvec'] = inputs['invec'] * 2.

    def compute_partials(self, inputs, partials):
        partials['outvec', 'invec'] = 2.0 * np.ones((self.options['arr_size'],))


class DistribCompSimple(om.ExplicitComponent):
    """Uses 2 procs but takes full input vars"""

    def initialize(self):
        self.options.declare('arr_size', types=int, default=10,
                             desc="Size of input and output vectors.")

    def setup(self):
        arr_size = self.options['arr_size']

        self.add_input('invec', np.ones(arr_size, float), distributed=True)
        self.add_output('outvec', np.ones(arr_size, float), distributed=True)

    def compute(self, inputs, outputs):
        if MPI and self.comm.size > 1:
            if self.comm.rank == 0:
                outvec = inputs['invec'] * 0.25
            else:
                outvec = inputs['invec'] * 0.5

            # now combine vecs from different processes
            both = np.zeros((2, len(outvec)))
            self.comm.Allgather(outvec, both)

            # add both together to get our output
            outputs['outvec'] = both[0, :] + both[1, :]
        else:
            outputs['outvec'] = inputs['invec'] * 0.75


class DistribInputComp(om.ExplicitComponent):
    """Uses all procs and takes input var slices"""

    def initialize(self):
        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

    def compute(self, inputs, outputs):
        if MPI:
            out = np.ascontiguousarray(outputs['outvec'])
            self.comm.Allgatherv(inputs['invec']*2.0,
                                 [out, self.sizes,
                                  self.offsets, MPI.DOUBLE])
            outputs.set_var('outvec', out)
        else:
            outputs['outvec'] = inputs['invec'] * 2.0

    def setup(self):
        comm = self.comm
        rank = comm.rank

        arr_size = self.options['arr_size']

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size, arr_size)

        # src_indices will be computed automatically
        self.add_input('invec', np.ones(self.sizes[rank], float), distributed=True)
        self.add_output('outvec', np.ones(arr_size, float), shape=np.int32(arr_size),
                        distributed=True)

class DistribOverlappingInputComp(om.ExplicitComponent):
    """Uses 2 procs and takes input var slices"""

    def initialize(self):
        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")
        self.options.declare('local_size', types=int, default=1,
                             desc="Local size of output vector.")

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

        arr_size = self.options['arr_size']
        local_size = self.options['local_size']

        self.add_output('outvec', np.zeros(arr_size, float), distributed=True)
        self.add_input('invec', np.ones(local_size, float), distributed=True)


class DistribInputDistribOutputComp(om.ExplicitComponent):
    """Uses 2 procs and takes input var slices."""

    def initialize(self):
        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']*2.0

    def setup(self):

        comm = self.comm
        rank = comm.rank

        arr_size = self.options['arr_size']

        sizes, offsets = evenly_distrib_idxs(comm.size, arr_size)
        self.sizes = sizes

        # don't set src_indices on the input and just use default behavior
        self.add_input('invec', np.ones(sizes[rank], float), distributed=True)
        self.add_output('outvec', np.ones(sizes[rank], float), distributed=True)


class DistribCompWithDerivs(om.ExplicitComponent):
    """Uses 2 procs and takes input var slices, but also computes partials"""

    def initialize(self):
        self.options.declare('arr_size', types=int, default=11,
                             desc="Size of input and output vectors.")

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']*2.0

    def compute_partials(self, inputs, J):
        sizes = self.sizes
        comm = self.comm
        rank = comm.rank

        J['outvec', 'invec'] = 2.0 * np.ones((sizes[rank],))

    def setup(self):

        comm = self.comm
        rank = comm.rank

        arr_size = self.options['arr_size']

        sizes, offsets = evenly_distrib_idxs(comm.size, arr_size)
        self.sizes = sizes

        # don't set src_indices on the input and just use default behavior
        self.add_input('invec', np.ones(sizes[rank], float), distributed=True)
        self.add_output('outvec', np.ones(sizes[rank], float), distributed=True)
        self.declare_partials('outvec', 'invec', rows=np.arange(0, sizes[rank]),
                                                 cols=np.arange(0, sizes[rank]))


class DistribInputDistribOutputDiscreteComp(DistribInputDistribOutputComp):

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        super().compute(inputs, outputs)
        discrete_outputs['disc_out'] = discrete_inputs['disc_in'] + 'bar'

    def setup(self):
        super().setup()
        self.add_discrete_input('disc_in', 'foo')
        self.add_discrete_output('disc_out', 'foobar')


class DistribNoncontiguousComp(om.ExplicitComponent):
    """Used in tests of non-contiguous input var slices"""

    def initialize(self):
        self.options.declare('size', types=int,
                             desc="Size of input and output vectors.")

    def compute(self, inputs, outputs):
        outputs['outvec'] = inputs['invec']*2.0

    def setup(self):
        size = self.options['size']

        self.add_input('invec', np.ones(size, float), distributed=True)
        self.add_output('outvec', np.ones(size, float), distributed=True)


class DistribGatherComp(om.ExplicitComponent):
    """Uses 2 procs gathers a distrib input into a full output"""

    def initialize(self):
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

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size, arr_size)

        # need to initialize the variable to have the correct local size
        # src_indices will be computed automatically
        self.add_input('invec', np.ones(self.sizes[rank], float), distributed=True)
        self.add_output('outvec', np.ones(arr_size, float), distributed=True)


class NonDistribGatherComp(om.ExplicitComponent):
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


@unittest.skipUnless(PETScVector is None, "Only runs when PETSc is not available")
class NOMPITests(unittest.TestCase):

    def test_distrib_idx_in_full_out(self):
        size = 11

        p = om.Problem()
        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')

        msg = "'C2' <class DistribInputComp>: Component contains distributed variables, " \
              "but there is no distributed vector implementation (MPI/PETSc) " \
              "available. The default non-distributed vectors will be used."

        with assert_warning(DistributedComponentWarning, msg):
            p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        p['C1.invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))


class DistribParaboloid(om.ExplicitComponent):

    def setup(self):
        if self.comm.rank == 0:
            ndvs = 3
        else:
            ndvs = 2

        self.add_input('w', val=1., distributed=True) # this will connect to a non-distributed IVC
        self.add_input('x', shape=ndvs, distributed=True) # this will connect to a distributed IVC

        self.add_output('y', shape=1, distributed=True) # all-gathered output, duplicated on all procs
        self.add_output('z', shape=ndvs, distributed=True) # distributed output
        self.declare_partials('y', 'x')
        self.declare_partials('y', 'w')
        self.declare_partials('z', 'x')

    def compute(self, inputs, outputs):
        x = inputs['x']
        local_y = np.sum((x-5)**2)
        y_g = np.zeros(self.comm.size)
        self.comm.Allgather(local_y, y_g)
        outputs['y'] = np.sum(y_g) + (inputs['w']-10)**2
        outputs['z'] = x**2

    def compute_partials(self, inputs, J):
        x = inputs['x']
        J['y', 'x'] = 2*(x-5)
        J['y', 'w'] = 2*(inputs['w']-10)
        J['z', 'x'] = np.diag(2*x)


@unittest.skipUnless(MPI, "MPI is required.")
class DistributedIO(unittest.TestCase):

    N_PROCS = 2

    def test_driver_metadata(self):
        p = om.Problem()
        comm = p.comm

        d_ivc = p.model.add_subsystem('distrib_ivc',
                                    om.IndepVarComp(distributed=True),
                                    promotes=['*'])

        # Sending different values to different ranks
        if comm.rank == 0:
            ndvs = 3
        else:
            ndvs = 2
        d_ivc.add_output('x', 2*np.ones(ndvs))

        ivc = p.model.add_subsystem('ivc',
                                    om.IndepVarComp(distributed=False),
                                    promotes=['*'])
        ivc.add_output('w', 2.0)
        p.model.add_subsystem('dp', DistribParaboloid(), promotes=['*'])

        p.model.add_design_var('x', lower=-100, upper=100)
        p.model.add_objective('y')
        p.setup()
        p.run_model()


        # Check the local size of the design variables on each proc
        dvs = p.model.get_design_vars()
        for name, meta in dvs.items():
            model_size = meta['size']
            self.assertEqual(model_size, ndvs)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPITests(unittest.TestCase):

    N_PROCS = 2

    def test_dist_to_nondist_err(self):
        size = 5
        p = om.Problem()
        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('x', np.ones(size)))
        model.add_subsystem("Cdist", DistribInputDistribOutputComp(arr_size=size))
        model.add_subsystem("Cserial", InOutArrayComp(arr_size=size))
        model.connect('indep.x', 'Cdist.invec')
        model.connect('Cdist.outvec', 'Cserial.invec', src_indices=om.slicer[:])
        p.setup()
        p.run_model()
        msg = "<model> <class Group>: Non-distributed variable 'Cserial.invec' has a distributed source, 'Cdist.outvec', so you must retrieve its value using 'get_remote=True'."
        with self.assertRaises(Exception) as cm:
            p['Cserial.invec']
        self.assertEqual(str(cm.exception), msg)

        with self.assertRaises(Exception) as cm:
            p.get_val('Cserial.invec')
        self.assertEqual(str(cm.exception), msg)

        with self.assertRaises(Exception) as cm:
            p.get_val('Cserial.invec', get_remote=False)
        self.assertEqual(str(cm.exception), msg)

    def test_dist_to_dist_get_remote_False_err(self):
        size = 5
        p = om.Problem()
        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('x', np.ones(size*2)))
        model.add_subsystem("Cdist", DistribInputDistribOutputComp(arr_size=size*2))
        if p.comm.rank == 0:
            inds = [0,1,2]
        else:
            inds = [3,4]
        model.add_subsystem("Cdist2", DistribInputDistribOutputComp(arr_size=size))
        model.connect('indep.x', 'Cdist.invec')
        model.connect('Cdist.outvec', 'Cdist2.invec', src_indices=inds)

        p.setup()
        p.run_model()
        msg = "<model> <class Group>: Can't retrieve distributed variable 'Cdist2.invec' because its src_indices reference entries from other processes. You can retrieve values from all processes using `get_val(<name>, get_remote=True)`."
        with self.assertRaises(Exception) as cm:
            p.get_val('Cdist2.invec', get_remote=False)
        self.assertEqual(str(cm.exception), msg)

    def test_distrib_full_in_out(self):
        size = 11

        p = om.Problem()
        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribCompSimple(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')

        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        p['C1.invec'] = np.ones(size, float) * 5.0

        p.run_model()

        np.testing.assert_allclose(C2._outputs['outvec'], np.ones(size, float)*7.5)

    def test_distrib_check_partials(self):
        # will produce uneven array sizes which we need for the test
        size = 11

        p = om.Problem()
        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        top.add_subsystem("C2", DistribCompWithDerivs(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')

        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        # run the model and check partials
        p.run_model()

        # this used to fail (bug #1279)
        assert_check_partials(p.check_partials(out_stream=None))

    def test_list_inputs_outputs(self):
        size = 11

        test = self

        class Model(om.Group):
            def setup(self):
                self.add_subsystem("C1", InOutArrayComp(arr_size=size))
                self.add_subsystem("C2", DistribCompSimple(arr_size=size))
                self.connect('C1.outvec', 'C2.invec')

            def configure(self):
                # verify list_inputs/list_outputs work in configure
                inputs = self.C2.list_inputs(shape=True, val=True, out_stream=None)
                outputs = self.C2.list_outputs(shape=True, val=True, out_stream=None)
                verify(inputs, outputs, full_size=size*2, loc_size=size, pathnames=False, rank=0)

                inputs = self.C2.list_inputs(shape=True, val=True, all_procs=True, out_stream=None)
                outputs = self.C2.list_outputs(shape=True, val=True, all_procs=True, out_stream=None)
                verify(inputs, outputs, full_size=size*2, loc_size=size, pathnames=False)

        p = om.Problem(Model())

        def verify(inputs, outputs, full_size, loc_size, in_vals=1., out_vals=1., pathnames=False, rank=None):
            inputs = sorted(inputs)
            outputs = sorted(outputs)

            with multi_proc_exception_check(p.comm):
                if rank is None or p.comm.rank == rank:
                    test.assertEqual(len(inputs), 1)
                    name, meta = inputs[0]
                    test.assertEqual(name, 'C2.invec' if pathnames else 'invec')
                    test.assertTrue(meta['shape'] == (loc_size,))
                    test.assertEqual(meta['val'].size, full_size)
                    test.assertTrue(all(meta['val'] == in_vals*np.ones(full_size)))

                    test.assertEqual(len(outputs), 1)
                    name, meta = outputs[0]
                    test.assertEqual(name, 'C2.outvec' if pathnames else 'outvec')
                    test.assertTrue(meta['shape'] == (loc_size,))
                    test.assertTrue(all(meta['val'] == out_vals*np.ones(full_size)))

        p.setup()

        # verify list_inputs/list_outputs work before final_setup
        inputs = p.model.C2.list_inputs(shape=True, val=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, val=True, out_stream=None)
        verify(inputs, outputs, size*2, size, pathnames=False, rank=0)

        inputs = p.model.C2.list_inputs(shape=True, val=True, all_procs=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, val=True, all_procs=True, out_stream=None)
        verify(inputs, outputs, size*2, size, pathnames=False)

        p.final_setup()

        p['C1.invec'] = np.ones(size, float) * 5.0

        # verify list_inputs/list_outputs work before run
        inputs = p.model.C2.list_inputs(shape=True, val=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, val=True, out_stream=None)
        verify(inputs, outputs, size*2, size, pathnames=False, rank=0)

        inputs = p.model.C2.list_inputs(shape=True, val=True, all_procs=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, val=True, all_procs=True, out_stream=None)
        verify(inputs, outputs, size*2, size, pathnames=False)

        p.run_model()

        # verify list_inputs/list_outputs work after run
        inputs = p.model.C2.list_inputs(shape=True, val=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, val=True, out_stream=None)
        verify(inputs, outputs, size*2, size, in_vals=10., out_vals=7.5, pathnames=False, rank=0)

        inputs = p.model.C2.list_inputs(shape=True, val=True, all_procs=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, val=True, all_procs=True, out_stream=None)
        verify(inputs, outputs, size*2, size, in_vals=10., out_vals=7.5, pathnames=False)

    def test_distrib_list_inputs_outputs(self):
        size = 11

        test = self

        def verify(inputs, outputs, in_vals=1., out_vals=1., pathnames=False, comm=None, final=True, rank=None):
            global_shape = (size, ) if final else NA

            inputs = sorted(inputs)
            outputs = sorted(outputs)

            with multi_proc_exception_check(comm):
                if comm is not None:
                    sizes, offsets = evenly_distrib_idxs(comm.size, size)
                    local_size = sizes[comm.rank]
                else:
                    local_size = size

                if rank is None or comm is None or rank == comm.rank:
                    test.assertEqual(len(inputs), 1)
                    name, meta = inputs[0]
                    test.assertEqual(name, 'C2.invec' if pathnames else 'invec')
                    test.assertEqual(meta['shape'], (local_size,))
                    test.assertEqual(meta['global_shape'], global_shape)
                    test.assertTrue(all(meta['val'] == in_vals*np.ones(size)))

                    test.assertEqual(len(outputs), 1)
                    name, meta = outputs[0]
                    test.assertEqual(name, 'C2.outvec' if pathnames else 'outvec')
                    test.assertEqual(meta['shape'], (local_size,))
                    test.assertEqual(meta['global_shape'], global_shape)
                    test.assertTrue(all(meta['val'] == out_vals*np.ones(size)))

        class Model(om.Group):
            def setup(self):
                self.add_subsystem("C1", InOutArrayComp(arr_size=size))
                self.add_subsystem("C2", DistribInputDistribOutputComp(arr_size=size))
                self.add_subsystem("C3", DistribGatherComp(arr_size=size))
                self.connect('C1.outvec', 'C2.invec')
                self.connect('C2.outvec', 'C3.invec')

            def configure(self):
                # verify list_inputs/list_outputs work in configure for distributed comp on rank 0 only
                inputs = self.C2.list_inputs(shape=True, global_shape=True, val=True, out_stream=None)
                outputs = self.C2.list_outputs(shape=True, global_shape=True, val=True, out_stream=None)
                verify(inputs, outputs, pathnames=False, comm=self.comm, final=False, rank=0)

                # verify list_inputs/list_outputs work in configure for distributed comp on all ranks
                inputs = self.C2.list_inputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
                outputs = self.C2.list_outputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
                verify(inputs, outputs, pathnames=False, comm=self.comm, final=False)

        p = om.Problem(Model())
        p.setup()
        p.final_setup()

        # verify list_inputs/list_outputs work before final_setup for distributed comp on rank 0 only
        inputs = p.model.C2.list_inputs(shape=True, global_shape=True, val=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, global_shape=True, val=True, out_stream=None)
        verify(inputs, outputs, pathnames=False, comm=p.comm, final=True, rank=0)

        # verify list_inputs/list_outputs work before final_setup for distributed comp on all ranks
        inputs = p.model.C2.list_inputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
        verify(inputs, outputs, pathnames=False, comm=p.comm, final=True)

        p['C1.invec'] = np.ones(size, float) * 5.0

        # verify list_inputs/list_outputs work before run for distributed comp on rank 0 only
        inputs = p.model.C2.list_inputs(shape=True, global_shape=True, val=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, global_shape=True, val=True, out_stream=None)
        verify(inputs, outputs, pathnames=False, comm=p.comm, final=True, rank=0)

        # verify list_inputs/list_outputs work before run for distributed comp on all ranks
        inputs = p.model.C2.list_inputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
        verify(inputs, outputs, pathnames=False, comm=p.comm, final=True)

        p.run_model()

        # verify list_inputs/list_outputs work after run for distributed comp on rank 0 only
        inputs = p.model.C2.list_inputs(shape=True, global_shape=True, val=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, global_shape=True, val=True, out_stream=None)
        verify(inputs, outputs, in_vals=10., out_vals=20., pathnames=False, comm=p.comm, final=True, rank=0)

        # verify list_inputs/list_outputs work after run for distributed comp on all ranks
        inputs = p.model.C2.list_inputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
        outputs = p.model.C2.list_outputs(shape=True, global_shape=True, val=True, all_procs=True, out_stream=None)
        verify(inputs, outputs, in_vals=10., out_vals=20., pathnames=False, comm=p.comm, final=True)

    def test_distrib_idx_in_full_out(self):
        size = 11

        p = om.Problem()
        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')

        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        p['C1.invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_distrib_1D_dist_output(self):
        size = 11

        p = om.Problem()

        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribInputComp(arr_size=size))
        top.add_subsystem("C3", om.ExecComp("y=x", x=np.zeros(size*p.comm.size),
                                                 y=np.zeros(size*p.comm.size)))

        comm = p.comm
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        start = offsets[rank]
        end = start + sizes[rank]

        top.connect('C1.outvec', 'C2.invec', src_indices=np.arange(start, end, dtype=int))
        top.connect('C2.outvec', 'C3.x', src_indices=om.slicer[:])

        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        p['C1.invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C2._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_distrib_idx_in_distrb_idx_out(self):
        # normal comp to distrib comp to distrb gather comp
        size = 3

        p = om.Problem()
        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        top.add_subsystem("C2", DistribInputDistribOutputComp(arr_size=size))
        C3 = top.add_subsystem("C3", DistribGatherComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec')
        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        p['C1.invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        self.assertTrue(all(C3._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_noncontiguous_idxs(self):
        # take even input indices in 0 rank and odd ones in 1 rank
        size = 11

        p = om.Problem()
        top = p.model
        comm = p.comm

        idxs = list(take_nth(p.comm.rank, comm.size, range(size)))

        C1 = top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribNoncontiguousComp(size=len(idxs)))
        C3 = top.add_subsystem("C3", DistribGatherComp(arr_size=size))
        top.connect('C1.outvec', 'C2.invec', src_indices=idxs)
        top.connect('C2.outvec', 'C3.invec')
        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        p['C1.invec'] = np.array(range(size), float)

        p.run_model()

        if MPI:
            self.assertTrue(all(C2._outputs['outvec'] == np.array(list(idxs), 'f')*4))

            full_list = list(take_nth(0, 2, range(size))) + list(take_nth(1, 2, range(size)))
            self.assertTrue(all(C3._outputs['outvec'] == np.array(full_list, 'f')*4))
        else:
            self.assertTrue(all(C2._outputs['outvec'] == C1._outputs['outvec']*2.))
            self.assertTrue(all(C3._outputs['outvec'] == C2._outputs['outvec']))

    def test_overlapping_inputs_idxs(self):
        # distrib comp with src_indices that overlap, i.e. the same
        # entries are distributed to multiple processes
        size = 11

        p = om.Problem()

        # need to initialize the input to have the correct local size
        if p.comm.rank == 0:
            local_size = 8
            start = 0
            end = 8
        else:
            local_size = 7
            start = 4
            end = 11

        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        C2 = top.add_subsystem("C2", DistribOverlappingInputComp(arr_size=size, local_size=local_size))
        top.connect('C1.outvec', 'C2.invec', src_indices=np.arange(start, end, dtype=int))
        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        input_vec = np.array(range(size, 0, -1), float)
        p['C1.invec'] = input_vec

        # C1 (an InOutArrayComp) doubles the input_vec
        check_vec = input_vec * 2

        p.run_model()

        np.testing.assert_allclose(C2._outputs['outvec'][:4], check_vec[:4]*2)
        np.testing.assert_allclose(C2._outputs['outvec'][8:], check_vec[8:]*2)

        # overlapping part should be double size of the rest
        np.testing.assert_allclose(C2._outputs['outvec'][4:8], check_vec[4:8]*4)

        np.testing.assert_allclose(p.get_val('C2.invec', get_remote=True),
                                   np.hstack((check_vec[0:8], check_vec[4:11])))

        dist_out = p.get_val('C2.outvec', get_remote=True)
        np.testing.assert_allclose(dist_out[:11], dist_out[11:])
        np.testing.assert_allclose(dist_out[:4], check_vec[:4] * 2)
        np.testing.assert_allclose(dist_out[8:11], check_vec[8:] * 2)
        np.testing.assert_allclose(dist_out[4:8], check_vec[4:8] * 4)

    def test_nondistrib_gather(self):
        # regular comp --> distrib comp --> regular comp.  last comp should
        # automagically gather the full vector without declaring src_indices
        size = 11

        p = om.Problem()
        top = p.model
        top.add_subsystem("C1", InOutArrayComp(arr_size=size))
        top.add_subsystem("C2", DistribInputDistribOutputComp(arr_size=size))
        C3 = top.add_subsystem("C3", NonDistribGatherComp(size=size))
        top.connect('C1.outvec', 'C2.invec')
        top.connect('C2.outvec', 'C3.invec', om.slicer[:])
        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        p['C1.invec'] = np.array(range(size, 0, -1), float)

        p.run_model()

        if MPI and self.comm.rank == 0:
            self.assertTrue(all(C3._outputs['outvec'] == np.array(range(size, 0, -1), float)*4))

    def test_auto_ivc_error(self):
        size = 2

        prob = om.Problem()
        prob.model.add_subsystem("C", DistribCompSimple(arr_size=size))

        with self.assertRaises(RuntimeError) as context:
            prob.setup()
            prob.final_setup()

        err_msg = str(context.exception).split(':')[-1]
        self.assertEqual(err_msg, 'Distributed component input "C.invec" is not connected.')

    def test_auto_ivc_error_promoted(self):
        size = 2

        prob = om.Problem()
        prob.model.add_subsystem("C", DistribCompSimple(arr_size=size), promotes=['*'])
        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        err_msg = str(context.exception).split(':')[-1]
        self.assertEqual(err_msg, 'Distributed component input "C.invec", promoted as "invec", is not connected.')

    def test_bad_distrib_connect(self):
        class Adder(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape_by_conn=True, distributed=True)
                self.add_output('x_sum', shape=1)

            def compute(self, inputs, outputs):
                outputs['x_sum'] = np.sum(inputs['x'])

        prob = om.Problem(name='bad_distrib_problem')
        ivc = prob.model.add_subsystem('ivc',om.IndepVarComp())
        ivc.add_output('x', val = np.ones(10), distributed=True)

        prob.model.add_subsystem('adder', Adder())

        prob.model.connect('ivc.x0','adder.x')
        prob.setup()
        try:
            prob.final_setup()
        except Exception as err:
            self.assertTrue(
                "\nCollected errors for problem 'bad_distrib_problem':"
                "\n   <model> <class Group>: Attempted to connect from 'ivc.x0' to 'adder.x', but "
                "'ivc.x0' doesn't exist. Perhaps you meant to connect to one of the following outputs: ['ivc.x']."
                "\n   <model> <class Group>: Failed to resolve shapes for ['adder.x']. To see the "
                "dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'." in str(err))
        else:
            self.fail("Exception expected.")


class NonParallelTests(unittest.TestCase):

    def test_dist_to_nondist_no_err(self):
        size = 5
        p = om.Problem()
        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('x', np.ones(size)))
        model.add_subsystem("Cdist", DistribInputDistribOutputComp(arr_size=size))
        model.add_subsystem("Cserial", InOutArrayComp(arr_size=size))
        model.connect('indep.x', 'Cdist.invec')
        model.connect('Cdist.outvec', 'Cserial.invec')
        p.setup()
        p.run_model()

        # When model with distributed comp is run on a single processor,
        # it is not required to use get_remote=True
        assert_near_equal(p['Cserial.invec'], [2, 2, 2, 2, 2])
        assert_near_equal(p.get_val('Cserial.invec'), [2, 2, 2, 2, 2])
        assert_near_equal(p.get_val('Cserial.invec', get_remote=False), [2, 2, 2, 2, 2])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProbRemoteTests(unittest.TestCase):
    """
    Mostly tests get_val for distributed vars.
    """

    N_PROCS = 4

    def test_prob_getval_dist_par(self):
        size = 3

        p = om.Problem()
        top = p.model
        par = top.add_subsystem('par', om.ParallelGroup())

        ivc = om.IndepVarComp()
        ivc.add_output('invec1', np.ones(size))
        ivc.add_output('invec2', np.ones(size))
        top.add_subsystem('P', ivc)
        top.connect('P.invec1', 'par.C1.invec')
        top.connect('P.invec2', 'par.C2.invec')

        par.add_subsystem("C1", DistribInputDistribOutputComp(arr_size=size))
        par.add_subsystem("C2", DistribInputDistribOutputComp(arr_size=size))

        p.setup()

        p['P.invec1'] = np.array([2, 1, 1], float)
        p['P.invec2'] = np.array([6, 3, 3], float)

        p.run_model()

        ans = p.get_val('par.C2.invec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([6, 3,3], dtype=float))
        ans = p.get_val('par.C2.outvec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([12, 6, 6], dtype=float))
        ans = p.get_val('par.C1.invec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([2, 1, 1], dtype=float))
        ans = p.get_val('par.C1.outvec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([4, 2, 2], dtype=float))

    def test_prob_getval_dist_par_disc(self):
        size = 3

        p = om.Problem()
        top = p.model
        par = top.add_subsystem('par', om.ParallelGroup())

        ivc = om.IndepVarComp()
        ivc.add_output('invec1', np.ones(size))
        ivc.add_output('invec2', np.ones(size))
        ivc.add_discrete_output('disc_in1', 'C1foo')
        ivc.add_discrete_output('disc_in2', 'C2foo')
        top.add_subsystem('P', ivc)
        top.connect('P.invec1', 'par.C1.invec')
        top.connect('P.invec2', 'par.C2.invec')
        top.connect('P.disc_in1', 'par.C1.disc_in')
        top.connect('P.disc_in2', 'par.C2.disc_in')

        C1 = par.add_subsystem("C1", DistribInputDistribOutputDiscreteComp(arr_size=size))
        C2 = par.add_subsystem("C2", DistribInputDistribOutputDiscreteComp(arr_size=size))

        p.setup()

        p['P.invec1'] = np.array([2, 1, 1], float)
        p['P.invec2'] = np.array([6, 3, 3], float)

        p.run_model()

        ans = p.get_val('par.C2.invec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([6, 3,3], dtype=float))
        ans = p.get_val('par.C2.outvec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([12, 6, 6], dtype=float))
        ans = p.get_val('par.C1.invec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([2, 1, 1], dtype=float))
        ans = p.get_val('par.C1.outvec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([4, 2, 2], dtype=float))

        if C1 in p.model.par._subsystems_myproc:
            ans = p.get_val('par.C1.disc_in', get_remote=False)
            self.assertEqual(ans, 'C1foo')
            ans = p.get_val('par.C1.disc_out', get_remote=False)
            self.assertEqual(ans, 'C1foobar')

        if C2 in p.model.par._subsystems_myproc:
            ans = p.get_val('par.C2.disc_in', get_remote=False)
            self.assertEqual(ans, 'C2foo')
            ans = p.get_val('par.C2.disc_out', get_remote=False)
            self.assertEqual(ans, 'C2foobar')

        ans = p.get_val('par.C1.disc_in', get_remote=True)
        self.assertEqual(ans, 'C1foo')
        ans = p.get_val('par.C2.disc_in', get_remote=True)
        self.assertEqual(ans, 'C2foo')
        ans = p.get_val('par.C1.disc_out', get_remote=True)
        self.assertEqual(ans, 'C1foobar')
        ans = p.get_val('par.C2.disc_out', get_remote=True)
        self.assertEqual(ans, 'C2foobar')

    def test_prob_getval_dist_disc(self):
        size = 14

        p = om.Problem()

        top = p.model

        ivc = om.IndepVarComp()
        ivc.add_output('invec', np.ones(size))
        ivc.add_discrete_output('disc_in', 'C1foo')
        top.add_subsystem('P', ivc)
        top.connect('P.invec', 'C1.invec')
        top.connect('P.disc_in', 'C1.disc_in')

        top.add_subsystem("C1", DistribInputDistribOutputDiscreteComp(arr_size=size))
        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        rank = p.comm.rank

        p['P.invec'] = np.array([[4, 3, 2, 1, 8, 6, 4, 2, 9, 6, 3, 12, 8, 4.0]])
        p['P.disc_in'] = 'boo'

        p.run_model()

        if rank == 0:
            ans = p.get_val('C1.invec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([4,3,2,1], dtype=float))
            ans = p.get_val('C1.outvec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([8,6,4,2], dtype=float))
        elif rank == 1:
            ans = p.get_val('C1.invec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([8,6,4,2], dtype=float))
            ans = p.get_val('C1.outvec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([16,12,8,4], dtype=float))
        elif rank == 2:
            ans = p.get_val('C1.invec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([9,6,3], dtype=float))
            ans = p.get_val('C1.outvec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([18,12,6], dtype=float))
        elif rank == 3:
            ans = p.get_val('C1.invec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([12,8,4], dtype=float))
            ans = p.get_val('C1.outvec', get_remote=False)
            np.testing.assert_allclose(ans, np.array([24,16,8], dtype=float))

        ans = p.get_val('C1.invec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([4,3,2,1,8,6,4,2,9,6,3,12,8,4], dtype=float))
        ans = p.get_val('C1.outvec', get_remote=True)
        np.testing.assert_allclose(ans, np.array([8,6,4,2,16,12,8,4,18,12,6,24,16,8], dtype=float))

        ans = p.get_val('C1.disc_in', get_remote=False)
        self.assertEqual(ans, 'boo')
        ans = p.get_val('C1.disc_in', get_remote=True)
        self.assertEqual(ans, 'boo')
        ans = p.get_val('C1.disc_out', get_remote=False)
        self.assertEqual(ans, 'boobar')
        ans = p.get_val('C1.disc_out', get_remote=True)
        self.assertEqual(ans, 'boobar')


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPIFeatureTests(unittest.TestCase):

    N_PROCS = 2

    def test_distribcomp_feature(self):

        size = 15

        model = om.Group()

        # Distributed component "C2" requires an IndepVarComp to supply inputs.
        model.add_subsystem("indep", om.IndepVarComp('x', np.zeros(size)))
        model.add_subsystem("C2", DistribComp(size=size))
        model.add_subsystem("C3", Summer(size=size))

        model.connect('indep.x', 'C2.invec')
        # to copy the full distributed output C2.outvec into C3.invec on all procs, we need
        # to specify src_indices=om.slicer[:]
        model.connect('C2.outvec', 'C3.invec', src_indices=om.slicer[:])

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('indep.x', np.ones(size))
        prob.run_model()

        assert_near_equal(prob.get_val('C2.invec'),
                          np.ones((8,)) if model.comm.rank == 0 else np.ones((7,)))
        assert_near_equal(prob.get_val('C2.outvec'),
                          2*np.ones((8,)) if model.comm.rank == 0 else -3*np.ones((7,)))
        assert_near_equal(prob.get_val('C3.sum'), -5.)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestGroupMPI(unittest.TestCase):
    N_PROCS = 2

    def test_promote_distrib(self):

        class MyComp(om.ExplicitComponent):
            def initialize(self):
                self.options.declare('size', types=int, default=1,
                                     desc="Size of input vector x.")

            def setup(self):
                self.add_input('x', np.ones(self.options['size']), distributed=True)
                self.add_output('y', 1.0, distributed=True)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.arange(5, dtype=float)),
                              promotes_outputs=['x'])

        # decide what parts of the array we want based on our rank
        if p.comm.rank == 0:
            idxs = [0, 1, 2]
        else:
            # use [3, -1] here rather than [3, 4] just to show that we
            # can use negative indices.
            idxs = [3, -1]

        p.model.add_subsystem('C1', MyComp(size=len(idxs)))
        p.model.promotes('C1', inputs=['x'], src_indices=idxs)

        p.setup()
        p.set_val('x', np.arange(5, dtype=float))
        p.run_model()

        # each rank holds the assigned portion of the input array
        assert_near_equal(p.get_val('C1.x', get_remote=False),
                         np.arange(3, dtype=float) if p.model.C1.comm.rank == 0 else
                         np.arange(3, 5, dtype=float))

        # the output in each rank is based on the local inputs
        assert_near_equal(p.get_val('C1.y', get_remote=False), 6. if p.model.C1.comm.rank == 0 else 14.)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribCheckMPI(unittest.TestCase):
    N_PROCS = 2

    def test_distrib_conn_check(self):
        class Serial2Distributed(om.ExplicitComponent):
            def setup(self):
                self.add_input("serial_in", shape=3)

                if self.comm.rank == 0:
                    self.add_output("dist_out", shape=3, distributed=True)
                else:
                    self.add_output("dist_out", shape=0, distributed=True)

            def compute(self, inputs, outputs):
                if self.comm.rank == 0:
                    outputs["dist_out"] = inputs["serial_in"]

        class DistributedSum(om.ExplicitComponent):
            def setup(self):
                self.add_output("sum", shape=1)

            def compute(self, inputs, outputs):
                outputs["sum"] = self.comm.bcast(sum(inputs["dist_in"]), root=0)

        class SumGroup(om.Group):
            def setup(self):
                self.add_subsystem("s2d", Serial2Distributed(), promotes_inputs=[("serial_in", "in")])
                self.add_subsystem("sum", DistributedSum(), promotes_outputs=["sum"])
                self.sum.add_input("dist_in", shape_by_conn=True, distributed=True)
                self.connect("s2d.dist_out", "sum.dist_in")

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("ivc", om.IndepVarComp("x", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('sum1', SumGroup())
        parallel.add_subsystem('sum2', SumGroup())

        model.connect("ivc.x", "parallel.sum1.in", src_indices=om.slicer[:3])
        model.connect("ivc.x", "parallel.sum2.in", src_indices=om.slicer[3:])

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val("parallel.sum1.sum", get_remote=True), 3.0)
        assert_near_equal(prob.get_val("parallel.sum2.sum", get_remote=True), 12.0)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestCompDistVarErrors(unittest.TestCase):
    """
    When a component declares variables in different orders on different ranks
    or declares variables on some ranks but not others.
    """

    N_PROCS = 4

    def test_missing_vars(self):
        class Comp1(om.ExplicitComponent):
            def setup(self):
                if self.comm.rank==0:
                    self.add_output('a', np.zeros((2,3)), distributed=True)
                elif self.comm.rank==1:
                    self.add_output('a', np.zeros((1,3)), distributed=True)
                else:
                    self.add_output('a', np.zeros((0,3)), distributed=True)

        class Comp2(om.ExplicitComponent):
            def setup(self):
                self.add_input('a', distributed=True, shape_by_conn=True)
                if self.comm.rank==0:
                    self.add_output('b', np.zeros(2), distributed=True)
                    self.add_output('c', np.zeros(2), distributed=True)
                else:
                    self.add_output('b', np.zeros(0), distributed=True)

        class Group1(om.Group):
            def setup(self):
                self.add_subsystem('comp1', Comp1(), promotes=['*'])
                self.add_subsystem('comp2', Comp2(), promotes=['*'])

        prob = om.Problem()
        prob.model = Group1()

        with self.assertRaises(Exception) as cm:
            prob.setup(mode='rev')

        self.assertEqual(str(cm.exception),
                         "'comp2' <class Comp2>: Variable 'c' exists on some ranks and not others. "
                         "A component must declare all variables in the same order on all ranks, "
                         "even if the size of the variable is 0 on some ranks.")

    def test_out_of_order_vars(self):
        class Comp1(om.ExplicitComponent):
            def setup(self):
                if self.comm.rank==0:
                    self.add_output('a', np.zeros((2,3)), distributed=True)
                elif self.comm.rank==1:
                    self.add_output('a', np.zeros((1,3)), distributed=True)
                else:
                    self.add_output('a', np.zeros((0,3)), distributed=True)

        class Comp2(om.ExplicitComponent):
            def setup(self):
                self.add_input('a', distributed=True, shape_by_conn=True)
                if self.comm.rank==0:
                    self.add_output('b', np.zeros(2), distributed=True)
                    self.add_output('c', np.zeros(2), distributed=True)
                else:
                    self.add_output('c', np.zeros(1), distributed=True)
                    self.add_output('b', np.zeros(1), distributed=True)

        class Group1(om.Group):
            def setup(self):
                self.add_subsystem('comp1', Comp1(), promotes=['*'])
                self.add_subsystem('comp2', Comp2(), promotes=['*'])

        prob = om.Problem()
        prob.model = Group1()

        with self.assertRaises(Exception) as cm:
            prob.setup(mode='rev')

        self.assertEqual(str(cm.exception),
                         "'comp2' <class Comp2>: Variables have not been declared in the same "
                         "order on all ranks. A component must declare all variables in the "
                         "same order on all ranks, even if the size of the variable is 0 on some ranks.")


class SourceComponent(om.ExplicitComponent):
    """
    A simple component that acts as the source for a distributed variable.
    """

    def setup(self):
        self.add_output("y", shape=10, distributed=True)

    def compute(self, inputs, outputs):
        num_nodes = self.options["num_nodes"]
        outputs["y"] = np.arange(num_nodes, dtype=float) * 2.0


class DistributedComponent(om.ExplicitComponent):
    """
    This component has a distributed input. Its shape is not defined here
    but is inferred from the source of the connection during the setup phase.
    """

    def setup(self):
        self.add_input("x", distributed=True, shape_by_conn=True)

        self.add_output("sum_x", distributed=True, copy_shape="x")

    def compute(self, inputs, outputs):
        outputs["sum_x"] = np.sum(inputs["x"])


# --- Assemble the Model ---
# Create a Group to contain and connect the components.
class MyModel(om.Group):
    def setup(self):
        self.add_subsystem("source_comp", SourceComponent(), promotes_outputs=["y"])
        self.add_subsystem("distributed_comp", DistributedComponent(), promotes_inputs=["x"])
        self.connect("y", "x")


class TestDistDynShapesListOutputs(unittest.TestCase):
    N_PROCS = 2

    def test_dist_dyn_shapes_list_outputs(self):
        # this test passes if it doesn't raise an exception
        prob = om.Problem(model=MyModel())
        prob.setup()
        prob.model.list_outputs()



if __name__ == '__main__':
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
