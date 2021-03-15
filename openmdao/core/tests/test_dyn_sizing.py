import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from openmdao.utils.mpi import MPI
if MPI:
    try:
        from openmdao.vectors.petsc_vector import PETScVector
    except ImportError:
        PETScVector = None


class L2(om.ExplicitComponent):
    """takes the 2 norm of the input"""

    def setup(self):
        self.add_input('vec', shape_by_conn=True)
        self.add_output('val', 0.0)

    def compute(self, inputs, outputs):
        outputs['val'] = np.linalg.norm(inputs['vec'])


class TestAdder(unittest.TestCase):
    def test_adder(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar_feature import SellarMDA

        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp('in', np.ones(10)), promotes=['*'])

        prob.model.add_subsystem('L2norm', L2())
        prob.model.connect('in', ['L2norm.vec'])
        prob.setup()
        prob.run_model()
        np.testing.assert_allclose(prob['L2norm.vec'], np.ones(10))


# This is based on passing size information through the system shown below
# in all tests C starts with the shape information

# +-----+
# |     |
# |  A  +-----+
# |     |     |
# +-----+     |
#         +---+---+
#         |       |
#         |   B   +-----+
#         |       |     |
#         +-------+     |
#                   +---+----+
#                   |        |
#                   |   C    +-----+
#                   |        |     |
#                   +--------+ +---+---+
#                              |       |
#                              |   D   +----+
#                              |       |    |
#                              +-------+    |
#                                        +--+----+
#                                        |       |
#                                        |   E   |
#                                        |       |
#                                        +-------+

class B(om.ExplicitComponent):

    def setup(self):
        self.add_input('in', copy_shape='out')
        self.add_output('out', shape_by_conn=True)

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class C(om.ExplicitComponent):

    def setup(self):
        self.add_input('in', shape=4)
        self.add_output('out', shape=9)

    def compute(self, inputs, outputs):
        outputs['out'] = np.arange(9)


class D(om.ExplicitComponent):

    def setup(self):
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class E(om.ExplicitComponent):

    def setup(self):
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class B_distrib(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        self.add_input('in', copy_shape='out')
        self.add_output('out', shape_by_conn=True)

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class C_distrib(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        if self.comm.rank == 0:
            self.add_input('in', shape=1, src_indices=np.arange(0,1, dtype=int))
        elif self.comm.rank == 1:
            self.add_input('in', shape=2, src_indices=np.arange(1,3, dtype=int))
        else:
            self.add_input('in', shape=0, src_indices=np.arange(3,3, dtype=int))

        self.add_output('out', shape=3)

    def compute(self, inputs, outputs):
        outputs['out'] = np.sum(inputs['in']) * (self.comm.rank + 1)


class D_distrib(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class TestPassSize(unittest.TestCase):
    def test_serial(self):
        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C())
        prob.model.connect('B.out', ['C.in'])

        prob.model.add_subsystem('D', D())
        prob.model.connect('C.out', ['D.in'])

        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        prob.setup()
        prob.run_model()
        self.assertEqual(prob.get_val('A.out').size ,4)
        self.assertEqual(prob.get_val('B.in').size ,4)
        self.assertEqual(prob.get_val('B.out').size ,4)

        self.assertEqual(prob.get_val('D.in').size ,9)
        self.assertEqual(prob.get_val('D.out').size ,9)
        self.assertEqual(prob.get_val('E.in').size ,9)

    def test_unresolved_err(self):
        prob = om.Problem()
        prob.model = om.Group()

        prob.model.add_subsystem('B', B())
        prob.model.connect('C.out', ['B.in'])

        prob.model.add_subsystem('C', B())
        prob.model.connect('B.out', ['C.in'])

        with self.assertRaises(Exception) as raises_cm:
            prob.setup()

        exception = raises_cm.exception

        msg = "<model> <class Group>: Failed to resolve shapes for ['B.in', 'B.out', 'C.in', 'C.out']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
        self.assertEqual(exception.args[0], msg)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestPassSizeDistributed(unittest.TestCase):

    N_PROCS = 3

    def test_serial_start(self):
        """the size information starts in the serial comonent C"""

        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B_distrib())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C())
        prob.model.connect('B.out', ['C.in'])

        prob.model.add_subsystem('D', D_distrib())
        prob.model.connect('C.out', ['D.in'])

        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        prob.setup()
        prob.run_model()

        # # check all global sizes
        self.assertEqual(prob.get_val('A.out', get_remote=True).size, 4)
        self.assertEqual(prob.get_val('B.in', get_remote=True).size, 4)
        self.assertEqual(prob.get_val('B.out', get_remote=True).size, 4)

        self.assertEqual(prob.get_val('D.in', get_remote=True).size, 9)
        self.assertEqual(prob.get_val('D.out', get_remote=True).size, 9)
        self.assertEqual(prob.get_val('E.in', get_remote=True).size, 9)

        # #check all local sizes
        nprocs = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank

        # evenly distribute the variable over the procs
        ave, res = divmod(4, nprocs)
        sizes_up = [ave + 1 if p < res else ave for p in range(nprocs)]
        size_up = sizes_up[rank]

        ave, res = divmod(9, nprocs)
        sizes_down = [ave + 1 if p < res else ave for p in range(nprocs)]
        size_down = sizes_down[rank]

        self.assertEqual(prob.get_val('A.out').size, 4)
        self.assertEqual(prob.get_val('B.in').size, size_up)
        self.assertEqual(prob.get_val('B.out').size, sizes_up[rank])

        self.assertEqual(prob.get_val('D.in').size, size_down)
        self.assertEqual(prob.get_val('D.out').size, sizes_down[rank])
        self.assertEqual(prob.get_val('E.in', get_remote=True).size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('E.out').size, 9)

        # test the output from running model
        self.assertEqual(np.sum(prob.get_val('E.out')), np.sum(np.arange(9)))

    def test_distributed_start(self):
        """the size information starts in the distributed comonent C"""

        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B_distrib())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C_distrib())
        prob.model.connect('B.out', ['C.in'])

        prob.model.add_subsystem('D', D_distrib())
        prob.model.connect('C.out', ['D.in'])

        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        prob.setup()
        prob.run_model()

        # # check all global sizes
        self.assertEqual(prob.get_val('A.out', get_remote=True).size, 3)
        self.assertEqual(prob.get_val('B.in', get_remote=True).size, 3)
        self.assertEqual(prob.get_val('B.out', get_remote=True).size, 3)

        self.assertEqual(prob.get_val('D.in', get_remote=True).size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('D.out', get_remote=True).size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('E.in', get_remote=True).size, 3*self.N_PROCS)

        # #check all local sizes
        rank = MPI.COMM_WORLD.rank
        if rank == 0:
            size_up = 1
        elif rank == 1:
            size_up = 2
        else:
            size_up = 0

        size_down = 3

        self.assertEqual(prob.get_val('A.out').size, 3)
        self.assertEqual(prob.get_val('B.in').size, size_up)
        self.assertEqual(prob.get_val('B.out').size, size_up)

        self.assertEqual(prob.get_val('D.in').size, size_down)
        self.assertEqual(prob.get_val('D.out').size, size_down)
        self.assertEqual(prob.get_val('E.in', get_remote=True).size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('E.out').size, 3*self.N_PROCS)

        # test the output from running model
        n = self.N_PROCS - 1
        np.testing.assert_allclose(prob.get_val('E.out'), np.array([1., 1., 1., 4., 4., 4., 0., 0., 0.]))


class ResizableComp(om.ExplicitComponent):
    # this is just a component that allows us to resize between setups
    def __init__(self, n_inputs=1, size=5, mult=2.):
        super().__init__()
        self.n_inputs = n_inputs
        self.size = size
        self.mult = mult

    def setup(self):
        for i in range(self.n_inputs):
            self.add_input(f"x{i+1}", val=np.ones(self.size))
            self.add_output(f"y{i+1}", val=np.ones(self.size))

    def compute(self, inputs, outputs):
        for i in range(self.n_inputs):
            outputs[f"y{i+1}"] = self.mult*inputs[f"x{i+1}"]


class DynShapeComp(om.ExplicitComponent):
    # component whose inputs and outputs are dynamically shaped
    def __init__(self, n_inputs=1):
        super().__init__()
        self.n_inputs = n_inputs

        for i in range(self.n_inputs):
            self.add_input(f"x{i+1}", shape_by_conn=True, copy_shape=f"y{i+1}")
            self.add_output(f"y{i+1}", shape_by_conn=True, copy_shape=f"x{i+1}")

    def compute(self, inputs, outputs):
        for i in range(self.n_inputs):
            outputs[f"y{i+1}"] = 2*inputs[f"x{i+1}"]


class DistribDynShapeComp(om.ExplicitComponent):
    # a distributed component whose inputs and outputs are dynamically shaped
    def __init__(self, n_inputs=1):
        super().__init__()
        self.n_inputs = n_inputs
        self.options['distributed'] = True

    def setup(self):
        for i in range(self.n_inputs):
            self.add_input(f"x{i+1}", shape_by_conn=True, copy_shape=f"y{i+1}")
            self.add_output(f"y{i+1}", shape_by_conn=True, copy_shape=f"x{i+1}")

    def compute(self, inputs, outputs):
        for i in range(self.n_inputs):
            outputs[f"y{i+1}"] = 2*inputs[f"x{i+1}"]


class DistribComp(om.ExplicitComponent):
    # a distributed component with inputs and outputs that are not dynamically shaped
    def __init__(self, global_size, n_inputs=2):
        super().__init__()
        self.n_inputs = n_inputs
        self.global_size = global_size
        self.options['distributed'] = True

    def setup(self):
        # evenly distribute the variable over the procs
        ave, res = divmod(self.global_size, self.comm.size)
        sizes = [ave + 1 if p < res else ave for p in range(self.comm.size)]

        for i in range(self.n_inputs):
            self.add_input(f"x{i+1}", val=np.ones(sizes[rank]))
            self.add_output(f"y{i+1}", val=np.ones(sizes[rank]))

    def compute(self, inputs, outputs):
        for i in range(self.n_inputs):
            outputs[f"y{i+1}"] = (self.comm.rank + 1)*inputs[f"x{i+1}"]


class DynShapeGroupSeries(om.Group):
    # strings together some number of components in series.
    # component type is determined by comp_class
    def __init__(self, n_comps, n_inputs, comp_class):
        super().__init__()
        self.n_comps = n_comps
        self.n_inputs = n_inputs
        self.comp_class = comp_class

        for icmp in range(1, self.n_comps + 1):
            self.add_subsystem(f"C{icmp}", self.comp_class(n_inputs=self.n_inputs))

        for icmp in range(1, self.n_comps):
            for i in range(1, self.n_inputs + 1):
                self.connect(f"C{icmp}.y{i}", f"C{icmp+1}.x{i}")


class DynShapeGroupConnectedInputs(om.Group):
    # contains some number of components with all of their matching inputs connected.
    # component type is determined by comp_class
    def __init__(self, n_comps, n_inputs, comp_class):
        super().__init__()
        self.n_comps = n_comps
        self.n_inputs = n_inputs
        self.comp_class = comp_class

        for icmp in range(1, self.n_comps + 1):
            self.add_subsystem(f"C{icmp}", self.comp_class(n_inputs=self.n_inputs),
                               promotes_inputs=['*'])


class TestDynShapes(unittest.TestCase):
    def test_baseline_series(self):
        # this is just a sized source and unsized sink, and we put a DynShapeGroupSeries in between them
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((4,2)))
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3, 2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x1'},
                                                  y2={'shape_by_conn': True, 'copy_shape': 'x2'}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*16)

    def test_copy_shape_out_out(self):
        # test copy_shape from output to output
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((2,3)))
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3, 2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                  y1={'copy_shape': 'y2'},
                                                  y2={'copy_shape': 'y1'}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        np.testing.assert_allclose(p['sink.y2'], np.ones((2,3))*16)

    def test_copy_shape_in_in(self):
        # test copy_shape from input to input
        # The fact that this case works is a bit of a surprise since comp.x1 and comp.x2 do not set
        # shape_by_conn, so you would expect them to be unresolvable, but they connect to dynamic
        # shaped vars that DO have shape_by_conn set.  Basically, if shape_by_conn is set on either
        # end of a connection when both vars are dynamically shaped, it's the same effect as if
        # both had set shape_by_conn since the shapes of any two connected vars must match.
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((2,3)))
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3, 2, DynShapeComp))
        p.model.add_subsystem('comp', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'copy_shape': 'x2'},
                                                  x2={'copy_shape': 'x1'},
                                                  y1={'shape_by_conn': True},
                                                  y2={'shape_by_conn': True}))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1=np.ones((2,3)),
                                                  x2=np.ones((2,3)),
                                                  y1=np.ones((2,3)),
                                                  y2=np.ones((2,3))))
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.model.connect('Gdyn.C3.y1', 'comp.x1')
        p.model.connect('Gdyn.C3.y2', 'comp.x2')
        p.model.connect('comp.y1', 'sink.x1')
        p.model.connect('comp.y2', 'sink.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*32)
        np.testing.assert_allclose(p['sink.y2'], np.ones((2,3))*32)

    def test_copy_shape_in_in_unresolvable(self):
        # test copy_shape from input to input
        # In this case, our dynamicaly shaped inputs that do copy_shape from other inputs are connected to
        # non-dynamically shaped outputs, and because they don't set shape_by_conn, they are unresolvable,
        # unlike the test above where they connected to dynamically shaped outputs.
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((2,3)))
        p.model.add_subsystem('comp', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'copy_shape': 'x2'},
                                                  x2={'copy_shape': 'x1'},
                                                  y1={'shape_by_conn': True},
                                                  y2={'shape_by_conn': True}))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1=np.ones((2,3)),
                                                  x2=np.ones((2,3)),
                                                  y1=np.ones((2,3)),
                                                  y2=np.ones((2,3))))
        p.model.connect('indep.x1', 'comp.x1')
        p.model.connect('indep.x2', 'comp.x2')
        p.model.connect('comp.y1', 'sink.x1')
        p.model.connect('comp.y2', 'sink.x2')
        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        msg = "<model> <class Group>: Failed to resolve shapes for ['comp.x1', 'comp.x2']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
        self.assertEqual(cm.exception.args[0], msg)

    def test_mismatched_dyn_shapes(self):
        # this is a sized source and sink, but their sizes are incompatible
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((4,2)))
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3, 2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1=np.ones((2,3)),
                                                  x2=np.ones((3,2)),
                                                  y1=np.ones((2,3)),
                                                  y2=np.ones((3,2))))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        with self.assertRaises(Exception) as cm:
            p.setup()

        msg = "<model> <class Group>: Shape mismatch,  (3, 2) vs. (4, 2) for variable 'sink.x2' during dynamic shape determination."
        self.assertEqual(str(cm.exception), msg)

    def test_baseline_conn_inputs(self):
        # this is a sized source and unsized sink, with a DynShapeGroupConnectedInputs between them
        # indep.x? connects to Gdyn.C?.x?
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))),
                                      promotes_outputs=['*'])
        indep.add_output('x2', val=np.ones((4,2)))
        p.model.add_subsystem('Gdyn', DynShapeGroupConnectedInputs(2, 2, DynShapeComp),
                              promotes_inputs=['*'])
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x1'},
                                                  y2={'shape_by_conn': True, 'copy_shape': 'x2'}))
        p.model.connect('Gdyn.C1.y1', 'sink.x1')
        p.model.connect('Gdyn.C2.y2', 'sink.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*4)
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*4)
        np.testing.assert_allclose(p['Gdyn.C1.y2'], np.ones((4,2))*2)  # unconnected dyn shaped output
        np.testing.assert_allclose(p['Gdyn.C2.y1'], np.ones((2,3))*2)  # unconnected dyn shaped output

    def test_resetup(self):
        # test that the dynamic sizing reflects any changes that occur prior to 2nd call to setup.
        p = om.Problem()
        ninputs = 1
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(2, ninputs, DynShapeComp))
        comp = p.model.add_subsystem('sink', ResizableComp(ninputs, 10, 3.))
        p.model.connect('Gdyn.C2.y1', 'sink.x1')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones(10)*12)

        # now change the size and setup again
        comp.size = 5
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones(5)*12)

    def test_cycle_fwd_rev(self):
        # now put the DynShapeGroupSeries in a cycle (sink.y2 feeds back into Gdyn.C1.x2). Sizes are known
        # at both ends of the model (the IVC and at the sink)
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3,2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1=np.ones((2,3)),
                                                  x2=np.ones((4,2)),
                                                  y1=np.ones((2,3)),
                                                  y2=np.ones((4,2))))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('sink.y2', 'Gdyn.C1.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*16)
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        # each time we run_model, the value of sink.y2 will be multiplied by 16
        # because of the feedback
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*256)

    def test_cycle_rev(self):
        # now put the DynShapeGroupSeries in a cycle (sink.y2 feeds back into Gdyn.C1.x2), but here,
        # only the sink outputs are known and inputs are coming from auto_ivcs.
        p = om.Problem()
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3,2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1=np.ones((2,3)),
                                                  x2=np.ones((4,2)),
                                                  y1=np.ones((2,3)),
                                                  y2=np.ones((4,2))))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('sink.y2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*16)
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        # each time we run_model, the value of sink.y2 will be multiplied by 16
        # because of the feedback
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*256)

    def test_cycle_unresolved(self):
        # now put the DynShapeGroupSeries in a cycle (sink.y2 feeds back into Gdyn.C1.x2), but here,
        # sink.y2 is unsized, so no var in the '2' loop can get resolved.
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3,2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x1'},
                                                  y2={'shape_by_conn': True, 'copy_shape': 'x2'}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('sink.y2', 'Gdyn.C1.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        msg = "<model> <class Group>: Failed to resolve shapes for ['Gdyn.C1.x2', 'Gdyn.C1.y2', 'Gdyn.C2.x2', 'Gdyn.C2.y2', 'Gdyn.C3.x2', 'Gdyn.C3.y2', 'sink.x2', 'sink.y2']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
        self.assertEqual(str(cm.exception), msg)

    def test_bad_copy_shape_name(self):
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x11'}))
        p.model.connect('indep.x1', 'sink.x1')
        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        msg = "<model> <class Group>: Can't copy shape of variable 'sink.x11'. Variable doesn't exist."
        self.assertEqual(str(cm.exception), msg)

    def test_unconnected_var_dyn_shape(self):
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  y1={'shape_by_conn': True}))
        p.model.connect('indep.x1', 'sink.x1')
        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        msg = "<model> <class Group>: 'shape_by_conn' was set for unconnected variable 'sink.y1'."
        self.assertEqual(str(cm.exception), msg)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDynShapes(unittest.TestCase):
    N_PROCS = 4

    def test_remote_distrib(self):
        # this test has remote distributed components (distributed comps under parallel groups)
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x1', shape_by_conn=True)

        par = p.model.add_subsystem('par', om.ParallelGroup())
        G1 = par.add_subsystem('G1', DynShapeGroupSeries(2,1, DistribDynShapeComp))
        G2 = par.add_subsystem('G2', DynShapeGroupSeries(2,1, DistribDynShapeComp))

        p.model.add_subsystem('sink', om.ExecComp(['y1=x1+x2'], shape=(5,)))
        p.model.connect('indep.x1', ['par.G1.C1.x1', 'par.G2.C1.x1'])
        p.model.connect('par.G1.C2.y1', 'sink.x1')
        p.model.connect('par.G2.C2.y1', 'sink.x2')

        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones(5)*8.)


class DynPartialsComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True, copy_shape='y')
        self.add_output('y', shape_by_conn=True, copy_shape='x')

    def setup_partials(self):
        size = self._get_var_meta('x', 'size')
        self.mat = np.eye(size) * 3.
        rng = np.arange(size)
        self.declare_partials('y', 'x', rows=rng, cols=rng, val=3.0)

    def compute(self, inputs, outputs):
        outputs['y'] = self.mat.dot(inputs['x'])


class TestDynShapeFeature(unittest.TestCase):
    def test_feature_fwd(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.core.tests.test_dyn_sizing import DynPartialsComp

        p = om.Problem()
        p.model.add_subsystem('indeps', om.IndepVarComp('x', val=np.ones(5)))
        p.model.add_subsystem('comp', DynPartialsComp())
        p.model.add_subsystem('sink', om.ExecComp('y=x',
                                                  x={'shape_by_conn': True, 'copy_shape': 'y'},
                                                  y={'shape_by_conn': True, 'copy_shape': 'x'}))
        p.model.connect('indeps.x', 'comp.x')
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        J = p.compute_totals(of=['sink.y'], wrt=['indeps.x'])
        assert_near_equal(J['sink.y', 'indeps.x'], np.eye(5)*3.)

    def test_feature_rev(sefl):
        import numpy as np
        import openmdao.api as om
        from openmdao.core.tests.test_dyn_sizing import DynPartialsComp

        p = om.Problem()
        p.model.add_subsystem('comp', DynPartialsComp())
        p.model.add_subsystem('sink', om.ExecComp('y=x', shape=5))
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        J = p.compute_totals(of=['sink.y'], wrt=['comp.x'])
        assert_near_equal(J['sink.y', 'comp.x'], np.eye(5)*3.)

    def test_feature_middle(self):
        import numpy as np
        import openmdao.api as om

        class PartialsComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=np.ones(5))
                self.add_output('y', val=np.ones(5))

            def setup_partials(self):
                self.mat = np.eye(5) * 3.
                rng = np.arange(5)
                self.declare_partials('y', 'x', rows=rng, cols=rng, val=3.0)

            def compute(self, inputs, outputs):
                outputs['y'] = self.mat.dot(inputs['x'])

        p = om.Problem()
        p.model.add_subsystem('comp', PartialsComp())
        p.model.add_subsystem('sink', om.ExecComp('y=x',
                                                  x={'shape_by_conn': True, 'copy_shape': 'y'},
                                                  y={'shape_by_conn': True, 'copy_shape': 'x'}))
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        J = p.compute_totals(of=['sink.y'], wrt=['comp.x'])
        assert_near_equal(J['sink.y', 'comp.x'], np.eye(5)*3.)


# following 4 classes are used in TestDistribDynShapeCombos
class ser1(om.ExplicitComponent):
    def setup(self):
        rank = self.comm.rank
        var_shape = 2 * rank

        # this component outputs all serial => * connections
        self.add_output("ser_ser_fwd", shape=var_shape, val=np.ones(var_shape))
        self.add_output("ser_ser_bwd", shape_by_conn=True)

        self.add_output("ser_par_fwd", shape=var_shape, val=np.ones(var_shape))
        self.add_output("ser_par_bwd", shape_by_conn=True)

    def compute(self, inputs, outputs):
        pass

class par1(om.ExplicitComponent):
    def setup(self):
        rank = self.comm.rank
        var_shape = 2 * rank
        self.options['distributed'] = True

        # this component outputs all parallel => * connections
        self.add_output("par_ser_fwd", shape=var_shape, val=np.ones(var_shape))
        self.add_output("par_ser_bwd", shape_by_conn=True)

        self.add_output("par_par_fwd", shape=var_shape, val=np.ones(var_shape))
        self.add_output("par_par_bwd", shape_by_conn=True)

    def compute(self, inputs, outputs):
        pass

class ser2(om.ExplicitComponent):
    def setup(self):
        rank = self.comm.rank
        var_shape = 2 * rank
        # dummy output
        self.add_output('foo_ser2', val=1.)

        # this component receives all * => serial connections
        self.add_input("ivc_ser_fwd", shape_by_conn=True)
        self.add_input("ivc_ser_bwd", shape=var_shape, val=np.ones(var_shape))

        self.add_input("ser_ser_fwd", shape_by_conn=True)
        self.add_input("ser_ser_bwd", shape=var_shape, val=np.ones(var_shape))

        self.add_input("par_ser_fwd", shape_by_conn=True)
        self.add_input("par_ser_bwd", shape=var_shape, val=np.ones(var_shape))

    def compute(self, inputs, outputs):
        pass

class par2(om.ExplicitComponent):
    def setup(self):
        rank = self.comm.rank
        var_shape = 2 * rank
        self.options['distributed'] = True

        # dummy output
        self.add_output('foo_par2', val=1.)

        # this component receives all * => parallel connections
        self.add_input("ivc_par_fwd", shape_by_conn=True)
        self.add_input("ivc_par_bwd", shape=var_shape, val=np.ones(var_shape))

        self.add_input("ser_par_fwd", shape_by_conn=True)
        self.add_input("ser_par_bwd", shape=var_shape, val=np.ones(var_shape))

        self.add_input("par_par_fwd", shape_by_conn=True)
        self.add_input("par_par_bwd", shape=var_shape, val=np.ones(var_shape))

    def compute(self, inputs, outputs):
        pass

@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDynShapeCombos(unittest.TestCase):
    """
    This will test the dynamic shaping on parallel runs with all of the possible
    combinations of connections and dynamic shaping "directions". In words, we have
    independent variable components, serial components, and parallel components.
    Here is a list of possible connections (ser: serial, par: parallel):

    ivc => ser
    ivc => par
    ser => ser
    ser => par
    par => ser
    par => par

    We can use dynamic shaping for all 6 of these connection types, and the information
    for each connection can either be propagated "forward"/"fwd" (the upstream output
    is explicitly size, the downstream input is shaped by conn), or "backward"/"bwd"
    (the downstream input shape is explicitly specified, upstream output is shaped
    by conn). With 6 connection types and 2 connection direction, this results in 12
    dynamically sized connections to be checked. In these checks, we want to make sure
    OpenMDAO has an "expected" behavior, where the local size of the parameters are preserved
    on each processor regardless of connection type.

    In this test, we have a single model with 5 components in this order:
    ivc:  Independent variable comp. This will only be connected to a parallel or serial
    group and only has outputs by design.
    ser1: Serial component. This will be connected to serial or parallel. Again, only outputs variables and no inputs
    par1: Parallel component. This will be connected to serial or parallel. Again, only outputs variables and no inputs
    ser2: Serial component to receive connections that only has inputs and no outputs (only a dummy output)
    par2: Parallel component to receive connections that only has inputs and no outputs (only a dummy output)

    The variable naming convention goes like type1_type2_dir:
    type1 is the upstream component with the output
    type2 is the downstream component with the input
    direction is the direction of information in the dynamic sizing (see above for fwd, bwd)

    With all this context, here is a table that lists what variables test what i/o:

    Connection:    Forward direction   Backward direction
    ivc => ser2       ivc_ser_fwd         ivc_ser_bwd
    ivc => par2       ivc_par_fwd         ivc_par_bwd
    ser1 => ser2      ser_ser_fwd         ser_ser_bwd
    ser1 => par2      ser_par_fwd         ser_par_bwd
    par1 => ser2      par_ser_fwd         par_ser_bwd
    par1 => par2      par_par_fwd         par_par_bwd

    The reason we have this tests is that the parallel tests above do not cover every possible combination
    to keep things a bit simple, we do the combination tests here, and other features of the dynamic
    sizing is tested above (chain connections, shape copies etc.). In this test, we just focus on the
    individual dynamic shape copies in parallel runs and do not worry about dependencies.

    we use 3 processors for this:
    proc0: will have a size of 0 on all i/o. we dont really need to set different sizes
    since each connection does its own dynamic sizing
    proc1: will have a size of 2 on all i/o
    proc2: will have a size of 4 on all i/o

    So the variable sizes simply need to be rank*2
    """

    N_PROCS = 3

    def test_dyn_shape_combos(self):

        rank = self.comm.rank
        var_shape = 2 * rank

        p = om.Problem()

        # build the ivc
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes=["*"])
        ivc.add_output('ivc_ser_fwd', shape=var_shape, val=np.ones(var_shape))
        ivc.add_output('ivc_ser_bwd', shape_by_conn=True)

        ivc.add_output('ivc_par_fwd', shape=var_shape, val=np.ones(var_shape))
        ivc.add_output('ivc_par_bwd', shape_by_conn=True)

        # add the other components
        p.model.add_subsystem('ser1', ser1(), promotes=["*"])
        p.model.add_subsystem('par1', par1(), promotes=["*"])
        p.model.add_subsystem('ser2', ser2(), promotes=["*"])
        p.model.add_subsystem('par2', par2(), promotes=["*"])

        # setup
        p.setup()

        p.run_model()

        # test all of the i/o sizes set by shape_by_conn

        # ivc => serial
        self.assertEqual(p.get_val('ser2.ivc_ser_fwd').size, var_shape)
        self.assertEqual(p.get_val('ivc.ivc_ser_bwd').size, var_shape)

        # ivc => parallel
        self.assertEqual(p.get_val('par2.ivc_par_fwd').size, var_shape)
        self.assertEqual(p.get_val('ivc.ivc_par_bwd').size, var_shape)

        # serial => serial
        self.assertEqual(p.get_val('ser2.ser_ser_fwd').size, var_shape)
        self.assertEqual(p.get_val('ser1.ser_ser_bwd').size, var_shape)

        # serial => parallel
        self.assertEqual(p.get_val('par2.ser_par_fwd').size, var_shape)
        self.assertEqual(p.get_val('ser1.ser_par_bwd').size, var_shape)

        # parallel => serial
        self.assertEqual(p.get_val('ser2.par_ser_fwd').size, var_shape)
        self.assertEqual(p.get_val('par1.par_ser_bwd').size, var_shape)

        # parallel => parallel
        self.assertEqual(p.get_val('par2.par_par_fwd').size, var_shape)
        self.assertEqual(p.get_val('par1.par_par_bwd').size, var_shape)


if __name__ == "__main__":
    unittest.main()
