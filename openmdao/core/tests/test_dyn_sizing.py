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
class Ser1(om.ExplicitComponent):
    def setup(self):
        # serial components' outputs must be the same size on all procs.
        # this is a serial output, so the same output is duplicated on all procs

        # this component outputs all serial => * connections

        # downstream
        self.add_output("ser_ser_down", shape=4, val=np.ones(4))  # (1)
        self.add_output("ser_par_down", shape=4, val=np.ones(4))  # (2)

        # upstream
        self.add_output("ser_ser_up", shape_by_conn=True)  # (5), size should be 4
        self.add_output("ser_par_up", shape_by_conn=True)  # (6), size should be 4

    def compute(self, inputs, outputs):
        # check the 2 upstream connections to this component
        # np.testing.assert_equal(outputs["ser_ser_up"].size, 4)  # (5)
        # np.testing.assert_equal(outputs["ser_par_up"].size, 4)  # (6)

        return

class Par1(om.ExplicitComponent):
    def setup(self):
        rank = self.comm.rank
        var_shape = 2 * rank
        self.options['distributed'] = True

        # parallel components' outputs are distributed, so for a general case,
        # they will have different sizes on different processors.

        # this component outputs all parallel => * connections

        # downstream
        self.add_output("par_ser_down", shape=var_shape, val=np.ones(var_shape))  # (3)
        self.add_output("par_par_down", shape=var_shape, val=np.ones(var_shape))  # (4)

        # upstream
        self.add_output("par_ser_up", shape_by_conn=True)  # (7), size should be 2*rank
        self.add_output("par_par_up", shape_by_conn=True)  # (8), size should be 2*rank

    def compute(self, inputs, outputs):
        # check the 2 upstream connections to this component
        # np.testing.assert_equal(outputs["par_ser_up"].size, self.comm.rank * 2)  # (7)
        # np.testing.assert_equal(outputs["par_par_up"].size, self.comm.rank * 2)  # (8)

        return

class Ser2(om.ExplicitComponent):
    def setup(self):
        rank = self.comm.size
        var_shape = 2 * rank

        # serial components' inputs can be coming from another serial component,
        # and in that case, they are duplicated across procs and have the same size
        # OR serial components' inputs can also be coming from a parallel component,
        # and in that case, the vector is distributed and can have varying size on each proc

        # this component receives all * => serial connections

        # downstream
        self.add_input("ser_ser_down", shape_by_conn=True)  # (1), size should be 4
        self.add_input("par_ser_down", shape_by_conn=True)  # (3), size should be 2*rank

        # upstream
        self.add_input("ser_ser_up", shape=4)  # (5)
        self.add_input("par_ser_up", shape=var_shape, val=np.ones(var_shape))  # (7)

        # dummy output so the component runs
        self.add_output('foo_ser2', val=1.)

    def compute(self, inputs, outputs):
        # check the 2 downstream connections to this component
        np.testing.assert_equal(inputs["ser_ser_down"].size, 4)  # (1)
        np.testing.assert_equal(inputs["par_ser_down"].size, self.comm.rank * 2)  # (3)

        return

class Par2(om.ExplicitComponent):
    def setup(self):
        rank = self.comm.rank
        var_shape = 2 * rank
        self.options['distributed'] = True

        # parallel components' inputs can be coming from another serial component,
        # and in that case, they are duplicated across procs and have the same size
        # OR parallel components' inputs can also be coming from a parallel component,
        # and in that case, the vector is distributed and can have varying size on each proc

        # this component receives all * => parallel connections

        # downstream
        self.add_input("ser_par_down", shape_by_conn=True)  # (2), size should be 4
        self.add_input("par_par_down", shape_by_conn=True)  # (4), size should be 2*rank

        # upstream
        self.add_input("ser_par_up", shape=4)
        self.add_input("par_par_up", shape=var_shape, val=np.ones(var_shape))

        # dummy output
        self.add_output('foo_par2', val=1.)

    def compute(self, inputs, outputs):
        # check the 2 downstream connections to this component
        np.testing.assert_equal(inputs["ser_par_down"].size, 4)  # (2)
        np.testing.assert_equal(inputs["par_par_down"].size, self.comm.rank * 2)  # (4)

        return

@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDynShapeCombos(unittest.TestCase):
    """
    This will test the dynamic shaping on parallel runs with all of the possible
    combinations of connections and dynamic shaping directions. The "downstream"
    or "down" direction is when an output is sized, and the input is shaped by
    connection. The "upstream" or "up" direction is when an input is sized, and
    the output connected to it is shaped by connection. In both of these directions,
    we need to check all possible combinations of serial and distributed components
    in parallel runs.

    Here is a list of possible connections (ser: serial, par: parallel):

    ser => ser
    ser => par
    par => ser
    par => par

    We can use dynamic shaping for all 4 of these connection types, and the information
    for each connection can either be propagated down or up.  With 4 connection types
    and 2 connection direction, this results in 8 dynamically sized connections to be
    checked. In these checks, we want to make sure OpenMDAO has an "expected" behavior,
    where the local size of the parameters are preserved on each processor regardless
    of connection type.

    In this test, we have a single model with 4 components in this order:

    ser1:   Serial component. This will be connected to serial or parallel.
            Only outputs variables and no inputs
    par1:   Parallel component. This will be connected to serial or parallel.
            Again, only outputs variables and no inputs

    ser2:   Serial component to receive connections that only has inputs and no outputs
            (only a dummy output)
    par2:   Parallel component to receive connections that only has inputs and no outputs
            (only a dummy output)

    The variable naming convention goes like type1_type2_dir:
    type1 is the component with the output
    type2 is the component with the input
    direction is the direction of information in the dynamic sizing (see above for down and up)

    With all this context, here is a table that lists what variables test what i/o:

    Connection:      Downstream direction    Upstream direction
    ser1 => ser2       ser_ser_down (1)        ser_ser_up (5)
    ser1 => par2       ser_par_down (2)        ser_par_up (6)
    par1 => ser2       par_ser_down (3)        par_ser_up (7)
    par1 => par2       par_par_down (4)        par_par_up (8)

    The parentheses after the connection name shows where we make that connection below

    The reason we have this tests is that the parallel tests above do not cover every possible combination.
    It is important to remember that we are not just testing a serial to serial connection here.
    We are testing all these connection types in a parallel run,
    which is different than the serial tests above

    In these tests, the output of the ser1 component will have the same size of 4 on all processors.
    This is because the serial components are expected to have the "same" output duplicated on all procs.
    The inputs to the ser2 component can have varying sizes on different processors.

    The outputs of the par1 component will have a size of rank*2 on each proc.
    We use 3 processor for these tests because we want to chekc 2 procs with different non-zero sizes
    and one proc that has a zero size for that particular output.
    """

    N_PROCS = 3
    # def setUp(self)

    def test_dyn_shape_combos(self):

        rank = MPI.COMM_WORLD.rank
        var_shape = 2 * rank

        p = om.Problem()

        # components with outputs
        p.model.add_subsystem('ser1', Ser1())
        p.model.add_subsystem('par1', Par1())

        # components with inputs
        p.model.add_subsystem('ser2', Ser2())
        p.model.add_subsystem('par2', Par2())

        # all downstream connections
        p.model.connect('ser1.ser_ser_down', 'ser2.ser_ser_down')  # (1)
        p.model.connect('ser1.ser_par_down', 'par2.ser_par_down')  # (2)
        p.model.connect('par1.par_ser_down', 'ser2.par_ser_down')  # (3)
        p.model.connect('par1.par_par_down', 'par2.par_par_down')  # (4)

        # all upstream connections
        p.model.connect('ser1.ser_ser_up', 'ser2.ser_ser_up')  # (5)
        p.model.connect('ser1.ser_par_up', 'par2.ser_par_up')  # (6)
        p.model.connect('par1.par_ser_up', 'ser2.par_ser_up')  # (7)
        p.model.connect('par1.par_par_up', 'par2.par_par_up')  # (8)

        p.setup()

        # we check the variable sizes in each component here
        p.run_model()

        # also check the i/o sizes here
        p.model.list_inputs(shape=True, all_procs=True, global_shape=False)
        p.model.list_outputs(shape=True, all_procs=True, global_shape=False)

        # test all of the i/o sizes set by shape_by_conn

        # all downstream:

        # serial => serial (1)
        # AY: this works both in compute above, and here
        self.assertEqual(p.get_val('ser2.ser_ser_down').size, 4)

        # serial => parallel (2)
        # TODO AY: this variable passes the tests above in compute, but get_val fails when this is uncommented
        # self.assertEqual(p.get_val('par2.ser_par_down').size, 4)

        # parallel => serial (3)
        # TODO AY: passes the tests above in compute. the get_val does not work on this parallel output. need get_remote=True
        # self.assertEqual(p.get_val('ser2.par_ser_down').size, var_shape)

        # parallel => parallel (4)
        # AY: this works both in compute above, and here
        self.assertEqual(p.get_val('par2.par_par_down').size, var_shape)


        # all upstream:

        # # serial => serial (5)
        # self.assertEqual(p.get_val('ser1.ser_ser_up').size, 4)

        # # serial => parallel (6)
        # self.assertEqual(p.get_val('ser1.ser_par_up').size, 4)

        # # parallel => serial (7)
        # self.assertEqual(p.get_val('par1.par_ser_up').size, 2)

        # # parallel => parallel (8)
        # self.assertEqual(p.get_val('par1.par_par_up').size, var_shape)


if __name__ == "__main__":
    unittest.main()
    # myclass = TestDistribDynShapeCombos()
    # myclass.test_dyn_shape_combos()

