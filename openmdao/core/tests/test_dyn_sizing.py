import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.warnings import OMDeprecationWarning

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
        """the size information starts in the serial component C"""

        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B_distrib())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C())
        prob.model.connect('B.out', ['C.in'], src_indices=om.slicer[:])

        prob.model.add_subsystem('D', D_distrib())
        prob.model.connect('C.out', ['D.in'])

        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        with self.assertRaises(RuntimeError) as cm:
            prob.setup()

        msg = "<model> <class Group>: dynamic sizing of serial input 'E.in' from distributed output 'D.out' is not supported."
        self.assertEquals(str(cm.exception), msg)

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

        with self.assertRaises(RuntimeError) as cm:
            prob.setup()

        msg = "<model> <class Group>: dynamic sizing of serial output 'A.out' from distributed input 'B.in' is not supported because not all B.in ranks are the same size (sizes=[1 2 0])."
        self.assertEquals(str(cm.exception), msg)

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

        # 'sink' has a defined shape and dyn shapes propagate in reverse from there.
        p.model.add_subsystem('sink', om.ExecComp(['y1=x1+x2'], shape=(8,)))
        p.model.connect('indep.x1', ['par.G1.C1.x1', 'par.G2.C1.x1'])
        p.model.connect('par.G1.C2.y1', 'sink.x1', src_indices=om.slicer[:])
        p.model.connect('par.G2.C2.y1', 'sink.x2', src_indices=om.slicer[:])

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        cname = 'G1' if p.model.comm.rank <= 1 else 'G2'
        msg = f"'par.{cname}.C1' <class DistribDynShapeComp>: Can't determine src_indices automatically for input 'par.{cname}.C1.x1'. They must be supplied manually."
        self.assertEqual(str(cm.exception), msg)


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


class DistCompDiffSizeKnownInput(om.ExplicitComponent):
    def setup(self):
        size = (self.comm.rank + 1) * 3
        self.add_input('x', val=np.random.random(size), distributed=True)

class DistCompKnownInput(om.ExplicitComponent):
    def setup(self):
        size = 3
        self.add_input('x', val=np.random.random(size), distributed=True)

    def compute(self, inputs, outputs):
        pass

class DistCompUnknownInput(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True, distributed=True)

    def compute(self, inputs, outputs):
        pass


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDynShapeCombos(unittest.TestCase):
    """
    This will test the dynamic shaping on parallel runs with all of the possible
    combinations of connections and dynamic shaping directions.

    Here is a list of possible connections:

    serial => serial
    serial => distributed
    distributed => serial
    distributed => distributed
    """

    N_PROCS = 3

    def test_ser_known_ser_unknown(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', val=np.random.random(2))
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                  x={'shape_by_conn': True},
                                                  y=np.zeros(2)))
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x'), p.get_val('comp.x'))

    def test_ser_unknown_ser_known(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                  x=np.random.random(2),
                                                  y=np.zeros(2)))
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x'), p.get_val('comp.x'))

    def test_ser_known_dist_unknown(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())

        indeps.add_output('x', val=np.random.random(2))
        p.model.add_subsystem('comp', DistCompUnknownInput())
        p.model.connect('indeps.x', 'comp.x')
        msg = "Connection between serial output 'indeps.x' and distributed input 'comp.x' is deprecated and will become an error in a future release."
        with assert_warning(OMDeprecationWarning, msg):
            p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x'), p.get_val('comp.x'))

    def test_ser_unknown_dist_known_err(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True)
        p.model.add_subsystem('comp', DistCompDiffSizeKnownInput())
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertEquals(cm.exception.args[0],
                          "<model> <class Group>: dynamic sizing of serial output 'indeps.x' from distributed input 'comp.x' is not supported because not all comp.x ranks are the same size (sizes=[3 6 9]).")

    def test_ser_unknown_dist_known(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True)
        p.model.add_subsystem('comp', DistCompKnownInput())
        p.model.connect('indeps.x', 'comp.x')
        msg = "Connection between serial output 'indeps.x' and distributed input 'comp.x' is deprecated and will become an error in a future release."
        with assert_warning(OMDeprecationWarning, msg):
            p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x'), p.get_val('comp.x'))

    def test_dist_known_ser_unknown(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', np.ones(3), distributed=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                  x={'shape_by_conn': True},
                                                  y={'copy_shape': 'x'}))
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertEquals(cm.exception.args[0],
                          "<model> <class Group>: dynamic sizing of serial input 'comp.x' from distributed output 'indeps.x' is not supported.")

    def test_dist_unknown_ser_known(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', distributed=True, shape_by_conn=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2', shape=3))
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertEquals(cm.exception.args[0],
                          "<model> <class Group>: Can't connect distributed output 'indeps.x' to serial input 'comp.x' without specifying src_indices.")

    def test_dist_known_dist_unknown(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        sizes = [3,0,5]
        indeps.add_output('x', np.random.random(sizes[MPI.COMM_WORLD.rank]), distributed=True)
        p.model.add_subsystem('comp', DistCompUnknownInput())
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x'), p.get_val('comp.x'))

    def test_dist_unknown_dist_known(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True, distributed=True)
        p.model.add_subsystem('comp', DistCompDiffSizeKnownInput())
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x'), p.get_val('comp.x'))

if __name__ == "__main__":
    unittest.main()
