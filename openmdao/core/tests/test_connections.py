""" Tests related to connecing inputs to outputs."""

import unittest
import numpy as np

from io import StringIO

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestConnections(unittest.TestCase):

    def setUp(self):
        self.setup_model(None, None)

    def setup_model(self, c1meta=None, c3meta=None):
        self.p = om.Problem()
        root = self.p.model

        if c1meta is None:
            c1meta = {}

        if c3meta is None:
            c3meta = {}

        self.G1 = root.add_subsystem("G1", om.Group())
        self.G2 = self.G1.add_subsystem("G2", om.Group())
        self.C1 = self.G2.add_subsystem("C1", om.ExecComp('y=x*2.0', **c1meta))
        self.C2 = self.G2.add_subsystem("C2", om.IndepVarComp('x', 1.0))

        self.G3 = root.add_subsystem("G3", om.Group())
        self.G4 = self.G3.add_subsystem("G4", om.Group())
        self.C3 = self.G4.add_subsystem("C3", om.ExecComp('y=x*2.0', **c3meta))
        self.C4 = self.G4.add_subsystem("C4", om.ExecComp('y=x*2.0'))

    def test_no_conns(self):
        self.p.setup()

        self.p['G1.G2.C1.x'] = 111.
        self.p['G3.G4.C3.x'] = 222.
        self.p['G3.G4.C4.x'] = 333.

        self.p.run_model()

        self.assertEqual(self.C1._inputs['x'], 111.)
        self.assertEqual(self.C3._inputs['x'], 222.)
        self.assertEqual(self.C4._inputs['x'], 333.)

    def test_pull_size_from_source(self):
        raise unittest.SkipTest("setting input size based on src size not supported yet")

        class Src(ExplicitComponent):

            def setup(self):

                self.add_input('x', 2.0)
                self.add_output('y1', np.zeros((3, )))
                self.add_output('y2', shape=((3, )))

            def solve_nonlinear(self, inputs, outputs, resids):
                x = inputs['x']

                outputs['y1'] = x * np.array([1.0, 2.0, 3.0])
                outputs['y2'] = x * np.array([1.0, 2.0, 3.0])

        class Tgt(ExplicitComponent):

            def setup(self):

                self.add_input('x1')
                self.add_input('x2')
                self.add_output('y1', 0.0)
                self.add_output('y2', 0.0)

            def solve_nonlinear(self, inputs, outputs, resids):
                x1 = inputs['x1']
                x2 = inputs['x2']

                outputs['y1'] = np.sum(x1)
                outputs['y2'] = np.sum(x2)

        p = om.Problem()
        p.model.add_subsystem('src', Src())
        p.model.add_subsystem('tgt', Tgt())

        p.model.connect('src.y1', 'tgt.x1')
        p.model.connect('src.y2', 'tgt.x2')

        p.setup()
        p.run_model()

        self.assertEqual(p['tgt.y1'], 12.0)
        self.assertEqual(p['tgt.y2'], 12.0)

    def test_pull_size_from_source_with_indices(self):
        raise unittest.SkipTest("setting input size based on src size not supported yet")

        class Src(ExplicitComponent):

            def setup(self):

                self.add_input('x', 2.0)
                self.add_output('y1', np.zeros((3, )))
                self.add_output('y2', shape=((3, )))
                self.add_output('y3', 3.0)

            def solve_nonlinear(self, inputs, outputs, resids):
                """ counts up. """

                x = inputs['x']

                outputs['y1'] = x * np.array([1.0, 2.0, 3.0])
                outputs['y2'] = x * np.array([1.0, 2.0, 3.0])
                outputs['y3'] = x * 4.0

        class Tgt(ExplicitComponent):

            def setup(self):

                self.add_input('x1')
                self.add_input('x2')
                self.add_input('x3')
                self.add_output('y1', 0.0)
                self.add_output('y2', 0.0)
                self.add_output('y3', 0.0)

            def solve_nonlinear(self, inputs, outputs, resids):
                """ counts up. """

                x1 = inputs['x1']
                x2 = inputs['x2']
                x3 = inputs['x3']

                outputs['y1'] = np.sum(x1)
                outputs['y2'] = np.sum(x2)
                outputs['y3'] = np.sum(x3)

        top = om.Problem()
        top.model.add_subsystem('src', Src())
        top.model.add_subsystem('tgt', Tgt())

        top.model.connect('src.y1', 'tgt.x1', src_indices=(0, 1))
        top.model.connect('src.y2', 'tgt.x2', src_indices=(0, 1))
        top.model.connect('src.y3', 'tgt.x3')

        top.setup()
        top.run_model()

        self.assertEqual(top['tgt.y1'], 6.0)
        self.assertEqual(top['tgt.y2'], 6.0)
        self.assertEqual(top['tgt.y3'], 8.0)

    def test_inp_inp_conn_no_src(self):
        raise unittest.SkipTest("no setup testing yet")
        self.p.model.connect('G3.G4.C3.x', 'G3.G4.C4.x')

        stream = StringIO()
        self.p.setup(out_stream=stream)

        self.p['G3.G4.C3.x'] = 999.
        self.assertEqual(self.p.model.G3.G4.C3._inputs['x'], 999.)
        self.assertEqual(self.p.model.G3.G4.C4._inputs['x'], 999.)

        content = stream.getvalue()
        self.assertTrue("The following parameters have no associated unknowns:\n"
                        "G1.G2.C1.x\nG3.G4.C3.x\nG3.G4.C4.x" in content)
        self.assertTrue("The following components have no connections:\n"
                        "G1.G2.C1\nG1.G2.C2\nG3.G4.C3\nG3.G4.C4\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)


class TestConnectionsPromoted(unittest.TestCase):

    def test_inp_inp_promoted_w_prom_src(self):
        p = om.Problem()
        root = p.model

        G1 = root.add_subsystem("G1", om.Group(), promotes=['x'])
        G2 = G1.add_subsystem("G2", om.Group(), promotes=['x'])
        G2.add_subsystem("C1", om.ExecComp('y=x*2.0'))
        G2.add_subsystem("C2", om.IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add_subsystem("G3", om.Group(), promotes=['x'])
        G4 = G3.add_subsystem("G4", om.Group(), promotes=['x'])
        C3 = G4.add_subsystem("C3", om.ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add_subsystem("C4", om.ExecComp('y=x*2.0'), promotes=['x'])

        p.setup()
        p.set_solver_print(level=0)

        # setting promoted name will set the value into the outputs, but will
        # not propagate it to the inputs. That will happen during run_model().
        p['x'] = 999.

        p.run_model()
        self.assertEqual(C3._inputs['x'], 999.)
        self.assertEqual(C4._inputs['x'], 999.)

    def test_inp_inp_promoted_w_explicit_src(self):
        p = om.Problem()
        root = p.model

        G1 = root.add_subsystem("G1", om.Group())
        G2 = G1.add_subsystem("G2", om.Group(), promotes=['x'])
        G2.add_subsystem("C1", om.ExecComp('y=x*2.0'))
        G2.add_subsystem("C2", om.IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add_subsystem("G3", om.Group())
        G4 = G3.add_subsystem("G4", om.Group(), promotes=['x'])
        C3 = G4.add_subsystem("C3", om.ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add_subsystem("C4", om.ExecComp('y=x*2.0'), promotes=['x'])

        p.model.connect('G1.x', 'G3.x')
        p.setup()
        p.set_solver_print(level=0)

        # setting promoted name will set the value into the outputs, but will
        # not propagate it to the inputs. That will happen during run_model().
        p['G1.x'] = 999.

        p.run_model()
        self.assertEqual(C3._inputs['x'], 999.)
        self.assertEqual(C4._inputs['x'], 999.)

    def test_overlapping_system_names(self):
        # This ensures that _setup_connections does not think g1 and g1a are the same system
        prob = om.Problem()
        model = prob.model

        g1 = model.add_subsystem('g1', om.Group())
        g1a = model.add_subsystem('g1a', om.Group())

        g1.add_subsystem('c', om.ExecComp('y=x'))
        g1a.add_subsystem('c', om.ExecComp('y=x'))

        model.connect('g1.c.y', 'g1a.c.x')
        model.connect('g1a.c.y', 'g1.c.x')

        prob.setup(check=True)


class TestConnectionsIndices(unittest.TestCase):

    def setUp(self):
        class ArrayComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('inp', val=np.ones((2)))
                self.add_input('inp1', val=0)
                self.add_output('out', val=np.zeros((2)))

            def compute(self, inputs, outputs):
                outputs['out'] = inputs['inp'] * 2.

        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output('blammo', val=3.)
        indep_var_comp.add_output('arrout', val=np.ones(5))

        prob = om.Problem()
        prob.model.add_subsystem('idvp', indep_var_comp)
        prob.model.add_subsystem('arraycomp', ArrayComp())

        self.prob = prob

    def test_bad_shapes(self):
        # Should not be allowed because the source and target shapes do not match
        self.prob.model.connect('idvp.blammo', 'arraycomp.inp')

        expected = "<model> <class Group>: The source and target shapes do not match or are " + \
                   "ambiguous for the connection 'idvp.blammo' to 'arraycomp.inp'. " + \
                   "The source shape is (1,) but the target shape is (2,)."

        try:
            self.prob.setup()
        except ValueError as err:
            self.assertEqual(str(err), expected)
        else:
            self.fail('Exception expected.')

        self.prob.model._raise_connection_errors = False

        with assert_warning(UserWarning, expected):
            self.prob.setup()

    def test_bad_length(self):
        # Should not be allowed because the length of src_indices is greater than
        # the shape of arraycomp.inp
        self.prob.model.connect('idvp.blammo', 'arraycomp.inp', src_indices=[0, 1, 0])

        expected = "<model> <class Group>: The source indices [0 1 0] do not specify a valid shape " + \
                   "for the connection 'idvp.blammo' to 'arraycomp.inp'. The target shape is " + \
                   "(2,) but indices are (3,)."

        try:
            self.prob.setup()
        except ValueError as err:
            self.assertEqual(str(err), expected)
        else:
            self.fail('Exception expected.')

        self.prob.model._raise_connection_errors = False

        with assert_warning(UserWarning, expected):
            self.prob.setup()

    def test_bad_value(self):
        # Should not be allowed because the index value within src_indices is outside
        # the valid range for the source
        self.prob.model.connect('idvp.arrout', 'arraycomp.inp1', src_indices=[100000])

        expected = "<model> <class Group>: The source indices do not specify a valid index " + \
                   "for the connection 'idvp.arrout' to 'arraycomp.inp1'. " + \
                   "Index '100000' is out of range for source dimension of size 5."

        try:
            self.prob.setup()
        except ValueError as err:
            self.assertEqual(str(err), expected)
        else:
            self.fail('Exception expected.')

        self.prob.model._raise_connection_errors = False

        with assert_warning(UserWarning, expected):
            self.prob.setup()

    def test_bad_value_bug(self):
        # Should not be allowed because the 2nd index value within src_indices is outside
        # the valid range for the source.  A bug prevented this from being checked.
        self.prob.model.connect('idvp.arrout', 'arraycomp.inp', src_indices=[0, 100000])

        expected = "<model> <class Group>: The source indices do not specify a valid index " + \
                   "for the connection 'idvp.arrout' to 'arraycomp.inp'. " + \
                   "Index '100000' is out of range for source dimension of size 5."

        try:
            self.prob.setup()
        except ValueError as err:
            self.assertEqual(str(err), expected)
        else:
            self.fail('Exception expected.')

        self.prob.model._raise_connection_errors = False

        with assert_warning(UserWarning, expected):
            self.prob.setup()


class TestShapes(unittest.TestCase):
    def test_connect_flat_array_to_row_vector(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', val=np.arange(10)))
        p.model.add_subsystem('C1',
                              om.ExecComp('y=dot(x, A)',
                                          x={'value': np.zeros((1, 10))},
                                          A={'value': np.eye(10)},
                                          y={'value': np.zeros((1, 10))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_near_equal(p['C1.y'], np.arange(10)[np.newaxis, :])

    def test_connect_flat_array_to_col_vector(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', val=np.arange(10)))
        p.model.add_subsystem('C1',
                              om.ExecComp('y=dot(A, x)',
                                          x={'value': np.zeros((10, 1))},
                                          A={'value': np.eye(10)},
                                          y={'value': np.zeros((10, 1))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_near_equal(p['C1.y'], np.arange(10)[:, np.newaxis])

    def test_connect_row_vector_to_flat_array(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', val=np.arange(10)[np.newaxis, :]))
        p.model.add_subsystem('C1', om.ExecComp('y=5*x',
                                                x={'value': np.zeros(10)},
                                                y={'value': np.zeros(10)}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_near_equal(p['C1.y'], 5 * np.arange(10))

    def test_connect_col_vector_to_flat_array(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', val=np.arange(10)[:, np.newaxis]))
        p.model.add_subsystem('C1', om.ExecComp('y=5*x',
                                                x={'value': np.zeros(10)},
                                                y={'value': np.zeros(10)}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_near_equal(p['C1.y'], 5 * np.arange(10))

    def test_connect_flat_to_3d_array(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', val=np.arange(10)))
        p.model.add_subsystem('C1', om.ExecComp('y=5*x',
                                                x={'value': np.zeros((1, 10, 1))},
                                                y={'value': np.zeros((1, 10, 1))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_near_equal(p['C1.y'], 5 * np.arange(10)[np.newaxis, :, np.newaxis])

    def test_connect_flat_nd_to_flat_nd(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x',
                                                       val=np.arange(10)[np.newaxis, :, np.newaxis,
                                                                         np.newaxis]))
        p.model.add_subsystem('C1', om.ExecComp('y=5*x',
                                                x={'value': np.zeros((1, 1, 1, 10))},
                                                y={'value': np.zeros((1, 1, 1, 10))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_near_equal(p['C1.y'],
                         5 * np.arange(10)[np.newaxis, np.newaxis, np.newaxis, :])

    def test_connect_incompatible_shapes(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', val=np.arange(10)[np.newaxis, :,
                                                                              np.newaxis, np.newaxis]))
        p.model.add_subsystem('C1', om.ExecComp('y=5*x',
                                                x={'value': np.zeros((5, 2))},
                                                y={'value': np.zeros((5, 2))}))
        p.model.connect('indep.x', 'C1.x')

        expected = "<model> <class Group>: The source and target shapes do not match or are " + \
                   "ambiguous for the connection 'indep.x' to 'C1.x'. The source shape is " + \
                   "(1, 10, 1, 1) but the target shape is (5, 2)."

        with self.assertRaises(Exception) as context:
            p.setup()

        self.assertEqual(str(context.exception), expected)

        p.model._raise_connection_errors = False

        with assert_warning(UserWarning, expected):
            p.setup()


class TestMultiConns(unittest.TestCase):

    def test_mult_conns(self):

        class SubGroup(om.Group):
            def setup(self):
                self.add_subsystem('c1', om.ExecComp('y = 2*x', x=np.ones(4), y=2*np.ones(4)),
                                   promotes=['y', 'x'])
                self.add_subsystem('c2', om.ExecComp('z = 2*y', y=np.ones(4), z=2*np.ones(4)),
                                   promotes=['z', 'y'])

        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 10*np.ones(4))
        indeps.add_output('y', np.ones(4))

        prob.model.add_subsystem('sub', SubGroup())

        prob.model.connect('x', 'sub.x')
        prob.model.connect('y', 'sub.y')

        expected = "<model> <class Group>: The following inputs have multiple connections: " + \
                   "sub.c2.y from ['indeps.y', 'sub.c1.y']"

        with self.assertRaises(Exception) as context:
            prob.setup()

        self.assertEqual(str(context.exception), expected)

        prob.model._raise_connection_errors = False

        with assert_warning(om.SetupWarning, expected):
            prob.setup()

    def test_mixed_conns_same_level(self):

        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 10*np.ones(4))

        # c2.y is implicitly connected to c1.y
        prob.model.add_subsystem('c1', om.ExecComp('y = 2*x', x=np.ones(4), y=2*np.ones(4)),
                                 promotes=['y'])
        prob.model.add_subsystem('c2', om.ExecComp('z = 2*y', y=np.ones(4), z=2*np.ones(4)),
                                 promotes=['y'])

        # make a second, explicit, connection to y (which is c2.y promoted)
        prob.model.connect('indeps.x', 'y')

        expected = "<model> <class Group>: Input 'c2.y' cannot be connected to 'indeps.x' " + \
                   "because it's already connected to 'c1.y'"

        with self.assertRaises(Exception) as context:
            prob.setup()
            prob.final_setup()

        self.assertEqual(str(context.exception), expected)

        prob.model._raise_connection_errors = False

        with assert_warning(UserWarning, expected):
            prob.setup()

    def test_auto_ivc_ambiguous_with_src_indices_msg(self):

        class TComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('src_idx', [0, 1])

            def setup(self):
                src = self.options['src_idx']
                self.add_input('x', shape=2, src_indices=src, val=-2038.0)
                self.add_output('y', shape=2)
                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                outputs['y'] = 2.0 * inputs['x']


        prob = om.Problem()
        model = prob.model

        prob.model.add_subsystem('c1', TComp(src_idx=[0, 1]), promotes_inputs=['x'])
        prob.model.add_subsystem('c2', TComp(src_idx=[2, 3]), promotes_inputs=['x'])
        prob.model.add_subsystem('d1', TComp(src_idx=[0, 1]), promotes_inputs=[('x', 'zz')])
        prob.model.add_subsystem('d2', TComp(src_idx=[1, 2]), promotes_inputs=[('x', 'zz')])

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        msg = "The following inputs ['c1.x', 'c2.x'] are defined using src_indices but the total source "
        msg += "size is undetermined.  You can specify the src size by setting 'val' or 'src_shape' in a call to set_input_defaults, or by adding an IndepVarComp as the source."

        err_msg = str(context.exception).split(':')[-1]
        self.assertEqual(err_msg, msg)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestConnectionsDistrib(unittest.TestCase):
    N_PROCS = 2

    def test_serial_mpi_error(self):
        # Should still catch the bad index when we are running under mpi with no distributed comps.
        # A bug formerly prevented this.
        class TestComp(om.ExplicitComponent):

            def initialize(self):
                self.options['distributed'] = False

            def setup(self):
                self.add_input('x', shape=2, src_indices=[1, 2], val=-2038.0)
                self.add_output('y', shape=1)
                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

            def compute_partials(self, inputs, J):
                J['y', 'x'] = np.ones((2,))

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', np.array([1.0, 3.0])))
        model.add_subsystem('c3', TestComp())
        model.connect("p1.x", "c3.x")

        rank = prob.comm.rank
        expected = f"Exception raised on rank {rank}: <model> <class Group>: The source indices do not specify a valid index " + \
                   "for the connection 'p1.x' to 'c3.x'. " + \
                   "Index '2' is out of range for source dimension of size 2."
        try:
            prob.setup()
        except Exception as err:
            self.assertEqual(str(err).splitlines()[-1], expected)
        else:
            self.fail('Exception expected.')

    def test_serial_mpi_error_flat(self):
        # Make sure the flat branch works too.
        class TestComp(om.ExplicitComponent):

            def initialize(self):
                self.options['distributed'] = False

            def setup(self):
                self.add_input('x', shape=2, src_indices=[1, 2], val=-2038.0, flat_src_indices=True)
                self.add_output('y', shape=1)
                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

            def compute_partials(self, inputs, J):
                J['y', 'x'] = np.ones((2,))

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', np.array([1.0, 3.0])))
        model.add_subsystem('c3', TestComp())
        model.connect("p1.x", "c3.x")

        rank = prob.comm.rank
        expected = f"Exception raised on rank {rank}: <model> <class Group>: The source indices do not specify a valid index " + \
                   "for the connection 'p1.x' to 'c3.x'. " + \
                   "Index '2' is out of range for source dimension of size 2."

        try:
            prob.setup()
        except Exception as err:
            self.assertEqual(str(err).splitlines()[-1], expected)
        else:
            self.fail('Exception expected.')

@unittest.skipUnless(MPI, "MPI is required.")
class TestConnectionsError(unittest.TestCase):
    N_PROCS = 2

    def test_incompatible_src_indices(self):
        class TestCompDist(om.ExplicitComponent):
        # this comp is distributed and forces PETScTransfer
            def initialize(self):
                self.options['distributed'] = True

            def setup(self):
                self.add_input('x', shape=2)
                self.add_output('y', shape=1)
                self.declare_partials('y', 'x', val=1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        class TestComp(om.ExplicitComponent):
            def initialize(self):
                self.options['distributed'] = False

            def setup(self):
                # read SRC_INDICES on each proc
                self.add_input('x', shape=2, src_indices=[1, 2], val=-2038.0)
                self.add_output('y', shape=1)
                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

            def compute_partials(self, inputs, J):
                J['y', 'x'] = np.ones((2,))

        prob = om.Problem()
        model = prob.model

        rank = prob.comm.rank

        if rank == 0:
            setval = np.array([2.0, 3.0])
        else:
            setval = np.array([10.0, 20.0])

        # no parallel or distributed comps, so default_vector is used (local xfer only)
        model.add_subsystem('p1', om.IndepVarComp('x', setval))
        model.add_subsystem('c3', TestComp())
        model.add_subsystem('c4', TestCompDist())
        model.connect("p1.x", "c3.x")
        model.connect("c3.y", "c4.x")

        with self.assertRaises(ValueError) as context:
            prob.setup(check=False, mode='fwd')
        self.assertEqual(str(context.exception),
                         f"Exception raised on rank {rank}: <model> <class Group>: The source indices do not specify a valid index for "
                         "the connection 'p1.x' to 'c3.x'. Index '2' is out of range for source "
                         "dimension of size 2.")


@unittest.skipUnless(MPI, "MPI is required.")
class TestConnectionsMPIBug(unittest.TestCase):
    N_PROCS = 2

    def test_bug_2d_src_indices(self):
        # This model gave an exception during setup.

        class Burn(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.arange(12))
                self.add_output('y', np.arange(12))

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] * 2.0

        class LinkageComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('in1', np.zeros((3, 2)))
                self.add_input('in2', np.zeros((3, 2)))
                self.add_output('out', np.zeros((3, 2)))

            def compute(self, inputs, outputs):
                outputs['out'] = 3 * inputs['in2'] - 2.5 * inputs['in1']

        class Phases(om.ParallelGroup):

            def setup(self):
                self.add_subsystem('burn1', Burn())
                self.add_subsystem('burn2', Burn())

        class Linkages(om.Group):

            def setup(self):
                self.add_subsystem('linkage', LinkageComp())

        class Traj(om.Group):

            def setup(self):
                self.add_subsystem('phases', Phases())
                self.add_subsystem('linkages', Linkages())

            def configure(self):
                self.connect('phases.burn1.y', 'linkages.linkage.in1', src_indices=np.array([[0, 3], [4, 6], [2, 1]]))
                self.connect('phases.burn2.y', 'linkages.linkage.in2', src_indices=np.array([[0, 3], [4, 6], [2, 1]]))

        prob = om.Problem(model=Traj())
        prob.setup()
        prob.run_model()


if __name__ == "__main__":
    unittest.main()
