import unittest
import sys

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
    def setup(self):
        self.add_input('in', copy_shape='out', distributed=True)
        self.add_output('out', shape_by_conn=True, distributed=True)

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class C_distrib(om.ExplicitComponent):
    def setup(self):
        if self.comm.rank == 0:
            self.add_input('in', shape=1, distributed=True)
        elif self.comm.rank == 1:
            self.add_input('in', shape=2, distributed=True)
        else:
            self.add_input('in', shape=0, distributed=True)

        self.add_output('out', shape=3, distributed=True)

    def compute(self, inputs, outputs):
        outputs['out'] = np.sum(inputs['in']) * (self.comm.rank + 1)


class D_distrib(om.ExplicitComponent):
    def setup(self):
        self.add_input('in', shape_by_conn=True, distributed=True)
        self.add_output('out', copy_shape='in', distributed=True)

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
        prob = om.Problem(name='unresolved_err')
        prob.model = om.Group()

        prob.model.add_subsystem('B', B())
        prob.model.connect('C.out', ['B.in'])

        prob.model.add_subsystem('C', B())
        prob.model.connect('B.out', ['C.in'])

        with self.assertRaises(Exception) as raises_cm:
            prob.setup()

        exception = raises_cm.exception

        self.assertEqual(exception.args[0],
            "\nCollected errors for problem 'unresolved_err':"
            "\n   <model> <class Group>: Failed to resolve shapes for "
            "['B.in', 'B.out', 'C.in', 'C.out']. To see the dynamic shape dependency graph, do "
            "'openmdao view_dyn_shapes <your_py_file>'.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestPassSizeDistributed(unittest.TestCase):

    N_PROCS = 3

    def test_serial_start(self):
        """the size information starts in the duplicated component C"""

        prob = om.Problem(name='serial_start')
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

        with self.assertRaises(Exception) as cm:
            prob.setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'serial_start':"
            "\n   <model> <class Group>: dynamic sizing of non-distributed input 'E.in' from distributed output 'D.out' is not supported."
            "\n   <model> <class Group>: Failed to resolve shapes for ['E.in', 'E.out']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
            "\n   <model> <class Group>: Can't connect distributed output 'D.out' to non-distributed input 'E.in' without specifying src_indices."
            "\n   <model> <class Group>: The source indices slice(None, None, 1) do not specify a valid shape for the connection 'B.out' to 'C.in'. The target shape is (4,) but indices are shape (12,).")

    def test_distributed_start(self):
        """the size information starts in the distributed component C"""

        prob = om.Problem(name='distributed_start')
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B_distrib())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C_distrib())
        if self.comm.rank == 0:
            prob.model.connect('B.out', ['C.in'], src_indices=np.arange(0,1, dtype=int))
        elif self.comm.rank == 1:
            prob.model.connect('B.out', ['C.in'], src_indices=np.arange(1,3, dtype=int))
        else:
            prob.model.connect('B.out', ['C.in'], src_indices=np.arange(3,3, dtype=int))

        prob.model.add_subsystem('D', D_distrib())
        prob.model.connect('C.out', ['D.in'])

        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        with self.assertRaises(RuntimeError) as cm:
            prob.setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'distributed_start':"
           "\n   <model> <class Group>: dynamic sizing of non-distributed output 'A.out' from distributed input 'B.in' is not supported because not all B.in ranks are the same size (sizes=[1 2 0])."
           "\n   <model> <class Group>: dynamic sizing of non-distributed input 'E.in' from distributed output 'D.out' is not supported."
           "\n   <model> <class Group>: Failed to resolve shapes for ['A.out', 'E.in', 'E.out']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
           "\n   'B' <class B_distrib>: Can't determine src_indices automatically for input 'B.in'. They must be supplied manually."
           "\n   <model> <class Group>: Can't connect distributed output 'D.out' to non-distributed input 'E.in' without specifying src_indices.")

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

    def setup(self):
        for i in range(self.n_inputs):
            self.add_input(f"x{i+1}", shape_by_conn=True, copy_shape=f"y{i+1}", distributed=True)
            self.add_output(f"y{i+1}", shape_by_conn=True, copy_shape=f"x{i+1}", distributed=True)

    def compute(self, inputs, outputs):
        for i in range(self.n_inputs):
            outputs[f"y{i+1}"] = 2*inputs[f"x{i+1}"]


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
    def test_simple(self):
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((4,2)))
        p.model.add_subsystem('C1', DynShapeComp(2))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x1'},
                                                  y2={'shape_by_conn': True, 'copy_shape': 'x2'}))
        p.model.connect('C1.y1', 'sink.x1')
        p.model.connect('C1.y2', 'sink.x2')
        p.model.connect('indep.x1', 'C1.x1')
        p.model.connect('indep.x2', 'C1.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*4)
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*4)

    def test_simple_compute_shape(self):
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((4,2)))
        p.model.add_subsystem('C1', DynShapeComp(2))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'shape_by_conn': True, 'compute_shape': lambda shapes: shapes['y1']},
                                                  x2={'shape_by_conn': True, 'compute_shape': lambda shapes: shapes['y2']},
                                                  y1={'shape_by_conn': True, 'compute_shape': lambda shapes: shapes['x1']},
                                                  y2={'shape_by_conn': True, 'compute_shape': lambda shapes: shapes['x2']}))
        p.model.connect('C1.y1', 'sink.x1')
        p.model.connect('C1.y2', 'sink.x2')
        p.model.connect('indep.x1', 'C1.x1')
        p.model.connect('indep.x2', 'C1.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*4)
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*4)

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
                                                  y2={'copy_shape': 'x1'}))
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
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((2,3)))
        p.model.add_subsystem('Gdyn', DynShapeGroupSeries(3, 2, DynShapeComp))
        p.model.add_subsystem('comp', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'copy_shape': 'x2'},
                                                  x2={'copy_shape': 'x1', 'shape_by_conn': True,},
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
        p = om.Problem(name='copy_shape_in_in_unresolvable')
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

        msg = "\nCollected errors for problem 'copy_shape_in_in_unresolvable':\n   <model> <class Group>: Failed to resolve shapes for ['comp.x1', 'comp.x2']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.\n   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection 'indep.x1' to 'comp.x1'. The source shape is (2, 3) but the target shape is None.\n   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection 'indep.x2' to 'comp.x2'. The source shape is (2, 3) but the target shape is None."
        self.assertEqual(cm.exception.args[0], msg)

    def test_mismatched_dyn_shapes(self):
        # this is a sized source and sink, but their sizes are incompatible
        p = om.Problem(name='mismatched_dyn_shapes')
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

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'mismatched_dyn_shapes':"
           "\n   <model> <class Group>: Shape mismatch, (4, 2) vs. (3, 2) for variables 'Gdyn.C2.x2' and 'Gdyn.C2.y2' during dynamic shape determination.")

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
        p = om.Problem(name='cycle_unresolved')
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
        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'cycle_unresolved':"
           "\n   <model> <class Group>: Failed to resolve shapes for "
           "['Gdyn.C1.x2', 'Gdyn.C1.y2', 'Gdyn.C2.x2', 'Gdyn.C2.y2', 'Gdyn.C3.x2', 'Gdyn.C3.y2', "
           "'sink.x2', 'sink.y2']. To see the dynamic shape dependency graph, do "
           "'openmdao view_dyn_shapes <your_py_file>'.")

    def test_bad_copy_shape_name(self):
        p = om.Problem(name='bad_copy_shape_name')
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x11'}))
        p.model.connect('indep.x1', 'sink.x1')
        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'bad_copy_shape_name':"
           "\n   <model> <class Group>: Can't copy shape of variable 'sink.x11'. Variable doesn't exist or is not continuous."
           "\n   <model> <class Group>: Failed to resolve shapes for ['sink.y1']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.")

    def test_unconnected_var_dyn_shape(self):
        p = om.Problem(name='unconnected_var_dyn_shape')
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  y1={'shape_by_conn': True}))
        p.model.connect('indep.x1', 'sink.x1')
        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'unconnected_var_dyn_shape':"
           "\n   <model> <class Group>: 'shape_by_conn' was set for unconnected variable 'sink.y1'."
           "\n   <model> <class Group>: Failed to resolve shapes for ['sink.y1']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDynShapes(unittest.TestCase):
    N_PROCS = 4

    def test_remote_distrib(self):
        # this test has remote distributed components (distributed comps under parallel groups)
        p = om.Problem(name='remote_distrib')
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

        self.assertTrue(
            "Collected errors for problem 'remote_distrib':\n"
            "   'par.G1.C1' <class DistribDynShapeComp>: Can't determine src_indices automatically for input 'par.G1.C1.x1'. They must be supplied manually.\n"
            "   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection 'indep.x1' to 'par.G1.C1.x1'. The source shape is (32,) but the target shape is (8,).\n"
            "   <model> <class Group>: The source indices slice(None, None, 1) do not specify a valid shape for the connection 'par.G1.C2.y1' to 'sink.x1'. The target shape is (8,) but indices are shape (16,).\n"
            "   <model> <class Group>: The source indices slice(None, None, 1) do not specify a valid shape for the connection 'par.G2.C2.y1' to 'sink.x2'. The target shape is (8,) but indices are shape (16,).\n"
            "   'par.G2.C1' <class DistribDynShapeComp>: Can't determine src_indices automatically for input 'par.G2.C1.x1'. They must be supplied manually.\n"
            "   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection 'indep.x1' to 'par.G2.C1.x1'. The source shape is (32,) but the target shape is (8,)."
           in str(cm.exception))


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

    def test_feature_rev(self):

        p = om.Problem()
        p.model.add_subsystem('comp', DynPartialsComp())
        p.model.add_subsystem('sink', om.ExecComp('y=x', shape=5))
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        J = p.compute_totals(of=['sink.y'], wrt=['comp.x'])
        assert_near_equal(J['sink.y', 'comp.x'], np.eye(5)*3.)

    def test_feature_middle(self):

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

    duplicated => duplicated
    duplicated => distributed
    distributed => duplicated
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

    def test_ser_unknown_dist_known_err(self):
        p = om.Problem(name='ser_unknown_dist_known_err')
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True)
        p.model.add_subsystem('comp', DistCompDiffSizeKnownInput())
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'ser_unknown_dist_known_err':"
           "\n   <model> <class Group>: dynamic sizing of non-distributed output 'indeps.x' from distributed input 'comp.x' is not supported because not all comp.x ranks are the same size (sizes=[3 6 9])."
           "\n   <model> <class Group>: Failed to resolve shapes for ['indeps.x']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
           "\n   'comp' <class DistCompDiffSizeKnownInput>: Can't determine src_indices automatically for input 'comp.x'. They must be supplied manually.")

    def test_dist_known_ser_unknown(self):
        p = om.Problem(name='dist_known_ser_unknown')
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', np.ones(3), distributed=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                  x={'shape_by_conn': True},
                                                  y={'copy_shape': 'x'}))
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'dist_known_ser_unknown':"
            "\n   <model> <class Group>: dynamic sizing of non-distributed input 'comp.x' from distributed output 'indeps.x' is not supported."
            "\n   <model> <class Group>: Failed to resolve shapes for ['comp.x', 'comp.y']. To see the dynamic shape dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
            "\n   <model> <class Group>: Can't connect distributed output 'indeps.x' to non-distributed input 'comp.x' without specifying src_indices.")

    def test_dist_unknown_ser_known(self):
        p = om.Problem(name='dist_unknown_ser_known')
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', distributed=True, shape_by_conn=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2', shape=3))
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertTrue(
            "\nCollected errors for problem 'dist_unknown_ser_known':"
            "\n   <model> <class Group>: Can't connect distributed output 'indeps.x' to "
            "non-distributed input 'comp.x' without specifying src_indices." in cm.exception.args[0])

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


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDynShapesEmptyError(unittest.TestCase):
    N_PROCS = 2

    def test_empty_error(self):
        # before the fix, this test raised an exception during setup

        # Input vector on root to be broadcast
        in_vec = np.arange(3)
        vec_len = len(in_vec)
        class BCastComp(om.ExplicitComponent):
            """
            Broadcast a vector from the root to all other processors.
            The output of this component is a serial vector.
            """

            def setup(self):
                # Distributed input, only one proc will have non-empty vector
                self.add_input('in_dist', shape_by_conn=True, distributed=True)

                # Serial Output (every processor has the same output vector)
                self.add_output('out_serial', shape=vec_len)

            def compute(self, inputs, outputs):
                # Root processor broadcasts values to everyone else
                outputs['out_serial'] = self.comm.bcast(inputs['in_dist'], root=0)

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == 'fwd':
                    if 'out_serial' in d_outputs:
                        if 'in_dist' in d_inputs:
                            d_outputs['out_serial'] += self.comm.bcast(d_inputs['in_dist'], root=0)

                else:  # 'rev'
                    if 'out_serial' in d_outputs:
                        if 'in_dist' in d_inputs:
                            if self.comm.rank == 0:
                                d_inputs['in_dist'] += d_outputs['out_serial']

        class Model(om.Group):
            """
            Simple group to hold problem components
            """

            def setup(self):
                # Create a distributed source for the distributed input.
                ivc = om.IndepVarComp()
                # Root processor gets input vector
                if self.comm.rank == 0:
                    ivc.add_output('x_dist', val=in_vec, shape=vec_len, distributed=True)
                # Every other proc has empty input
                else:
                    ivc.add_output('x_dist', shape=0, distributed=True)

                self.add_subsystem("indep", ivc)
                self.add_subsystem("bcast", BCastComp())

                self.connect('indep.x_dist', 'bcast.in_dist')

                self.add_design_var('indep.x_dist')
                self.add_constraint('bcast.out_serial', lower=0.0)

        prob = om.Problem(Model())
        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()


class DynShpComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', units='ft', shape_by_conn=True)
        self.add_output('y', copy_shape='x', units='ft')

    def compute(self, inputs, outputs):
        outputs['y'] = 3. * inputs['x']


class PGroup(om.Group):

    def setup(self):
        self.add_subsystem('comp1', DynShpComp(), promotes_inputs=['x'])
        self.add_subsystem('comp2', DynShpComp(), promotes_inputs=['x'])

    def configure(self):
        self.set_input_defaults('x', src_shape=(2, ))


class TestDynShapesWithInputConns(unittest.TestCase):
    # this tests the retrieval of shape info from a set_input_defaults call during
    # dynamic shape determination, which happens *before* group input defaults have
    # been fully processed.
    def test_group_input_defaults(self):
        prob = om.Problem()
        prob.model.add_subsystem('sub', PGroup())

        prob.setup()

        prob['sub.x'] = np.ones(2) * 7.
        prob.run_model()

        assert_near_equal(prob['sub.comp1.y'], np.ones(2) * 21.)
        assert_near_equal(prob['sub.comp2.y'], np.ones(2) * 21.)

    def test_shape_from_conn_input(self):
        prob = om.Problem()
        sub = prob.model.add_subsystem('sub', om.Group())
        comp1 = sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'shape_by_conn': True}, y={'copy_shape': 'x'}),
                                  promotes_inputs=['x'])
        comp2 = sub.add_subsystem('comp2', om.ExecComp('y=3*x', x=np.ones(2), y=np.zeros(2)),
                                  promotes_inputs=['x'])

        prob.setup()

        prob['sub.x'] = np.ones(2) * 7.
        prob.run_model()

        assert_near_equal(prob['sub.comp1.y'], np.ones(2) * 21.)
        assert_near_equal(prob['sub.comp2.y'], np.ones(2) * 21.)

    def test_shape_from_conn_input_mismatch(self):
        prob = om.Problem(name='shape_from_conn_input_mismatch')
        sub = prob.model.add_subsystem('sub', om.Group())
        comp1 = sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'shape_by_conn': True}, y={'copy_shape': 'x'}),
                                  promotes_inputs=['x'])
        comp2 = sub.add_subsystem('comp2', om.ExecComp('y=3*x', x=np.ones(2), y=np.zeros(2)),
                                  promotes_inputs=['x'])
        comp3 = sub.add_subsystem('comp3', om.ExecComp('y=3*x', x=np.ones(3), y=np.zeros(3)),
                                  promotes_inputs=['x'])

        with self.assertRaises(Exception) as cm:
            prob.setup()

        # just make sure we still get a clear error msg

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'shape_from_conn_input_mismatch':"
           "\n   <model> <class Group>: Shape of input 'sub.comp3.x', (3,), doesn't match shape (2,).")

    def test_shape_from_conn_input_mismatch_group_inputs(self):
        prob = om.Problem(name='shape_from_conn_input_mismatch_group_inputs')
        sub = prob.model.add_subsystem('sub', om.Group())
        comp1 = sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'shape_by_conn': True}, y={'copy_shape': 'x'}),
                                  promotes_inputs=['x'])
        comp2 = sub.add_subsystem('comp2', om.ExecComp('y=3*x', x=np.ones(2), y=np.zeros(2)),
                                  promotes_inputs=['x'])

        sub.set_input_defaults('x', src_shape=(3, ))

        with self.assertRaises(Exception) as cm:
            prob.setup()

        # just make sure we still get a clear error msg

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'shape_from_conn_input_mismatch_group_inputs':"
           "\n   <model> <class Group>: Shape of input 'sub.comp2.x', (2,), doesn't match shape (3,).")




class MMP(om.ExplicitComponent):
    def setup(self):
        self.add_input('M', shape_by_conn=True)
        self.add_input('N', shape_by_conn=True)
        self.add_output('out', compute_shape=lambda shapes: (shapes['M'][0], shapes['N'][1]))

    def compute(self, inputs, outputs):
        outputs['out'] = np.dot(inputs['M'], inputs['N'])


class TestComputeShape(unittest.TestCase):
    def test_mvp(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('M', val=np.random.random((3, 2)))
        indep.add_output('N', val=np.random.random((2, 8)))
        indep.add_output('O', val=np.random.random((8, 5)))
        indep.add_output('P', val=np.random.random((5, 7)))


        model.add_subsystem('C1', MMP())
        model.add_subsystem('C2', MMP())
        model.add_subsystem('C3', MMP())

        model.connect('indep.M', 'C1.M')
        model.connect('indep.N', 'C1.N')
        model.connect('indep.O', 'C2.M')
        model.connect('indep.P', 'C2.N')
        model.connect('C1.out', 'C3.M')
        model.connect('C2.out', 'C3.N')

        p.setup()
        p.run_model()

        self.assertEqual(model.C3._outputs['out'].shape, (3, 7))



@unittest.skipUnless(MPI and  PETScVector and sys.version_info >= (3, 9), "MPI, PETSc, and python 3.9+ are required.")
class TestLambdaPickle(unittest.TestCase):
    N_PROCS = 2

    def test_lambda_compute_shape(self):
        # this test passes if it doesn't raise an exception
        class Comp1(om.ExplicitComponent):
            def setup(self):
                self.add_output('a', np.zeros((2,3)), distributed=True)

        class Comp2(om.ExplicitComponent):
            def setup(self):
                self.add_input('a', distributed=True, shape_by_conn=True)
                self.add_output('b', compute_shape=lambda shapes: (shapes['a'][1],), distributed=True)

        class Group1(om.Group):
            def setup(self):
                self.add_subsystem('comp1', Comp1(), promotes=['*'])
                self.add_subsystem('comp2', Comp2(), promotes=['*'])

        class Group2(om.ParallelGroup):
            def setup(self):
                self.add_subsystem('group11', Group1())
                self.add_subsystem('group12', Group1())

        prob = om.Problem()
        prob.model.add_subsystem('par', Group2())
        prob.setup(mode='rev')
        prob.run_model()



if __name__ == "__main__":
    unittest.main()
