import unittest
import sys

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warnings

from openmdao.utils.mpi import MPI
if MPI:
    try:
        from openmdao.vectors.petsc_vector import PETScVector
    except ImportError:
        PETScVector = None
else:
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

        prob.model.add_subsystem('indeps', om.IndepVarComp('in', np.ones(10)), promotes=['*'])

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
            prob.final_setup()

        exception = raises_cm.exception

        self.assertEqual(exception.args[0],
            "\nCollected errors for problem 'unresolved_err':"
            "\n   <model> <class Group>: Failed to resolve shapes for "
            "['B.in', 'B.out', 'C.in', 'C.out']. To see the dynamic shapes dependency graph, do "
            "'openmdao view_dyn_shapes <your_py_file>'.")

    def test_err_msg_with_other_input(self):
        class CGInertia(om.ExplicitComponent):

            def setup(self):

                # This shape_by_conn is missing a source shape. An error is correctly
                # raised for this.
                self.add_input('Mission:FUEL_MASS', shape_by_conn=True, units="lbm")
                self.add_output('aircraft:CG', compute_shape=lambda shapes: (
                    shapes['Mission:FUEL_MASS'][0], 3))

                # However, when you have one more normal input, the error is short-circuited,
                # and setup2 bombs out while building a graph.
                # You can try with this line commented to see the correct error.
                self.add_input('aircraft:air_conditioning:mass', val=0.0, units="lbm")


        prob = om.Problem(name='dyn_err_check')
        prob.model.add_subsystem("cg", CGInertia(), promotes=["*"])
        prob.setup()
        with self.assertRaises(Exception) as cm:
            prob.final_setup()

        self.assertEqual(cm.exception.args[0],
                         "\nCollected errors for problem 'dyn_err_check':"
                         "\n   <model> <class Group>: Failed to resolve shapes for ['cg.Mission:FUEL_MASS', 'cg.aircraft:CG']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
                         "\n   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection '_auto_ivc.v0' to 'cg.Mission:FUEL_MASS'. The source shape is (1,) but the target shape is None.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestPassSizeDistributed(unittest.TestCase):

    N_PROCS = 3

    def test_serial_start_err(self):
        """the size information starts in the duplicated component C"""

        prob = om.Problem(name='serial_start_err')
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
            prob.final_setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'serial_start_err':"
            "\n   <model> <class Group>: Input 'C.in' has src_indices so the shape of connected output 'B.out' cannot be determined."
            "\n   <model> <class Group>: dynamic sizing of non-distributed input 'E.in' from distributed output 'D.out' without src_indices is not supported."
            "\n   <model> <class Group>: Failed to resolve shapes for ['A.out', 'B.in', 'B.out', 'E.in', 'E.out']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
            "\n   <model> <class Group>: Can't connect distributed output 'D.out' to non-distributed input 'E.in' without specifying src_indices.")

    def test_distributed_start_err(self):
        """the size information starts in the distributed component C"""

        prob = om.Problem(name='distributed_start_err')
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B_distrib())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C_distrib())
        if prob.comm.rank == 0:
            prob.model.connect('B.out', ['C.in'], src_indices=np.arange(0,1, dtype=int))
        elif prob.comm.rank == 1:
            prob.model.connect('B.out', ['C.in'], src_indices=np.arange(1,3, dtype=int))
        else:
            prob.model.connect('B.out', ['C.in'], src_indices=np.arange(3,3, dtype=int))

        prob.model.add_subsystem('D', D_distrib())
        prob.model.connect('C.out', ['D.in'])

        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        with self.assertRaises(RuntimeError) as cm:
            prob.setup()
            prob.final_setup()

        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'distributed_start_err':"
            "\n   <model> <class Group>: Input 'C.in' has src_indices so the shape of connected output 'B.out' cannot be determined."
            "\n   <model> <class Group>: dynamic sizing of non-distributed input 'E.in' from distributed output 'D.out' without src_indices is not supported."
            "\n   <model> <class Group>: Failed to resolve shapes for ['A.out', 'B.in', 'B.out', 'E.in', 'E.out']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
            "\n   <model> <class Group>: When connecting 'B.out' to 'C.in': index 0 is out of bounds for source dimension of size 0."
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
    def __init__(self, n_inputs=1, src_indices=None):
        super().__init__()
        self.n_inputs = n_inputs
        self.src_indices = src_indices

        for i in range(self.n_inputs):
            self.add_input(f"x{i+1}", shape_by_conn=True, copy_shape=f"y{i+1}")
            self.add_output(f"y{i+1}", shape_by_conn=self.src_indices is None, copy_shape=f"x{i+1}")

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
            self.declare_partials(of=f"y{i+1}", wrt=f"x{i+1}")

    def compute(self, inputs, outputs):
        for i in range(self.n_inputs):
            outputs[f"y{i+1}"] = 2*inputs[f"x{i+1}"]

    def compute_partials(self, inputs, partials):
        for i in range(self.n_inputs):
            size = self._var_sizes['input'][self.comm.rank][self._var_allprocs_abs2idx[f"{self.pathname}.x{i+1}"]]
            partials[f"y{i+1}", f"x{i+1}"] = 2*np.eye(size)


class SeriesGroup(om.Group):
    # strings together some number of components in series.
    # component type is determined by comp_class
    def __init__(self, n_comps, n_inputs, comp_class, src_indices=None, flat_src_indices=None):
        super().__init__()
        self.n_comps = n_comps
        self.n_inputs = n_inputs
        self.comp_class = comp_class
        self.src_indices = src_indices
        self.flat_src_indices = flat_src_indices
        assert(src_indices is None or len(src_indices) == n_inputs)

    def setup(self):
        kwargs = {
            'n_inputs': self.n_inputs,
        }
        if self.src_indices is not None:
            kwargs['src_indices'] = self.src_indices
        for icmp in range(1, self.n_comps + 1):
            self.add_subsystem(f"C{icmp}", self.comp_class(**kwargs))

        for icmp in range(1, self.n_comps):
            for i in range(1, self.n_inputs + 1):
                if self.src_indices is None or self.src_indices[i-1] is None:
                    src_indices = None
                else:
                    src_indices = self.src_indices[i-1]
                self.connect(f"C{icmp}.y{i}", f"C{icmp+1}.x{i}", src_indices=src_indices,
                             flat_src_indices=self.flat_src_indices)


class DynShapeGroupConnectedInputs(om.Group):
    # contains some number of components with all of their matching inputs connected.
    # component type is determined by comp_class
    def __init__(self, n_comps, n_inputs, comp_class):
        super().__init__()
        self.n_comps = n_comps
        self.n_inputs = n_inputs
        self.comp_class = comp_class

    def setup(self):
        for icmp in range(1, self.n_comps + 1):
            self.add_subsystem(f"C{icmp}", self.comp_class(n_inputs=self.n_inputs),
                               promotes_inputs=['*'])


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
        # this is just a sized source and unsized sink, and we put a SeriesGroup in between them
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((4,2)))
        p.model.add_subsystem('Gdyn', SeriesGroup(3, 2, DynShapeComp))
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

    def test_baseline_series_src_indices(self):
        # this is just a sized source and unsized sink, and we put a SeriesGroup in between them
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones(6)))
        indep.add_output('x2', val=np.ones(8))
        p.model.add_subsystem('Gdyn', SeriesGroup(2, 2, DynShapeComp,
                                                          src_indices=[om.slicer[:-2], om.slicer[2:]]))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x1'},
                                                  y2={'shape_by_conn': True, 'copy_shape': 'x2'}))
        p.model.connect('Gdyn.C2.y1', 'sink.x1')
        p.model.connect('Gdyn.C2.y2', 'sink.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones(4)*8)
        np.testing.assert_allclose(p['sink.y2'], np.ones(6)*8)

    def test_copy_shape_out_out(self):
        # test copy_shape from output to output
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((2,3)))
        p.model.add_subsystem('Gdyn', SeriesGroup(3, 2, DynShapeComp))
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
        p.model.add_subsystem('Gdyn', SeriesGroup(3, 2, DynShapeComp))
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
            p.final_setup()

        msg = "\nCollected errors for problem 'copy_shape_in_in_unresolvable':\n   <model> <class Group>: Failed to resolve shapes for ['comp.x1', 'comp.x2']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.\n   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection 'indep.x1' to 'comp.x1'. The source shape is (2, 3) but the target shape is None.\n   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection 'indep.x2' to 'comp.x2'. The source shape is (2, 3) but the target shape is None."
        self.assertEqual(cm.exception.args[0], msg)

    def test_mismatched_dyn_shapes_err(self):
        # this is a sized source and sink, but their sizes are incompatible
        p = om.Problem(name='mismatched_dyn_shapes_err')
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        indep.add_output('x2', val=np.ones((4,2)))
        p.model.add_subsystem('Gdyn', SeriesGroup(3, 2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1=np.ones((2,3)),
                                                  x2=np.ones((3,2)),
                                                  y1=np.ones((2,3)),
                                                  y2=np.ones((3,2))))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.setup()
        with self.assertRaises(Exception) as cm:
            p.run_model()

        msg = ("'Gdyn.C2' <class DynShapeComp>: Failed to set value of 'y2': could not broadcast input array from shape (4,2) into shape (3,2).")

        self.assertEqual(cm.exception.args[0], msg)

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
        p.model.add_subsystem('Gdyn', SeriesGroup(2, ninputs, DynShapeComp))
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
        # now put the SeriesGroup in a cycle (sink.y2 feeds back into Gdyn.C1.x2). Sizes are known
        # at both ends of the model (the IVC and at the sink)
        p = om.Problem()
        # p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('Gdyn', SeriesGroup(3,2, DynShapeComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1=np.ones((2,3)),
                                                  x2=np.ones((4,2)),
                                                  y1=np.ones((2,3)),
                                                  y2=np.ones((4,2))))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('sink.y2', 'Gdyn.C1.x2')
        # p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.setup()
        p.set_val('Gdyn.C1.x1', np.ones((2,3)))
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*16)
        p.run_model()
        np.testing.assert_allclose(p['sink.y1'], np.ones((2,3))*16)
        # each time we run_model, the value of sink.y2 will be multiplied by 16
        # because of the feedback
        np.testing.assert_allclose(p['sink.y2'], np.ones((4,2))*256)

    def test_cycle_rev(self):
        # now put the SeriesGroup in a cycle (sink.y2 feeds back into Gdyn.C1.x2), but here,
        # only the sink outputs are known and inputs are coming from auto_ivcs.
        p = om.Problem()
        p.model.add_subsystem('Gdyn', SeriesGroup(3,2, DynShapeComp))
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
        # now put the SeriesGroup in a cycle (sink.y2 feeds back into Gdyn.C1.x2), but here,
        # sink.y2 is unsized, so no var in the '2' loop can get resolved.
        p = om.Problem(name='cycle_unresolved')
        p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('Gdyn', SeriesGroup(3,2, DynShapeComp))
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
            p.final_setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'cycle_unresolved':"
           "\n   <model> <class Group>: Failed to resolve shapes for "
           "['Gdyn.C1.x2', 'Gdyn.C1.y2', 'Gdyn.C2.x2', 'Gdyn.C2.y2', 'Gdyn.C3.x2', 'Gdyn.C3.y2', "
           "'sink.x2', 'sink.y2']. To see the dynamic shapes dependency graph, do "
           "'openmdao view_dyn_shapes <your_py_file>'.")

    def test_bad_copy_shape_name(self):
        p = om.Problem(name='bad_copy_shape_name')
        p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  y1={'shape_by_conn': True, 'copy_shape': 'x11'}))
        p.model.connect('indep.x1', 'sink.x1')
        p.setup()
        expected_warnings = (
            (om.OpenMDAOWarning, "<model> <class Group>: 'shape_by_conn' was set for unconnected variable 'sink.y1'."),
            (om.OpenMDAOWarning, "<model> <class Group>: Can't copy shape of variable 'sink.x11'. Variable doesn't exist or is not continuous.")
        )

        with assert_warnings(expected_warnings):
            p.model._setup_dynamic_properties()

        with self.assertRaises(Exception) as cm:
            p.final_setup()

        self.assertEqual(cm.exception.args[0],
                         "\nCollected errors for problem 'bad_copy_shape_name':"
                         "\n   <model> <class Group>: Failed to resolve shapes for ['sink.y1']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.")

    def test_unconnected_var_dyn_shape(self):
        p = om.Problem(name='unconnected_var_dyn_shape')
        p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                  y1={'shape_by_conn': True}))
        p.model.connect('indep.x1', 'sink.x1')
        p.setup()

        expected_warnings = (
            (om.OpenMDAOWarning, "<model> <class Group>: 'shape_by_conn' was set for unconnected variable 'sink.y1'."),
        )

        with assert_warnings(expected_warnings):
            p.model._setup_dynamic_properties()

        with self.assertRaises(Exception) as cm:
            p.final_setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'unconnected_var_dyn_shape':"
           "\n   <model> <class Group>: Failed to resolve shapes for ['sink.y1']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestRemoteDistribDynShapes(unittest.TestCase):
    N_PROCS = 4

    def _build_model(self, solve_dyn_fwd):
        # this test has remote distributed components (distributed comps under parallel groups)
        p = om.Problem(name='remote_distrib_err')
        indep = p.model.add_subsystem('indep', om.IndepVarComp())
        if solve_dyn_fwd:
            indep.add_output('x1', val=np.random.random(4))
        else:
            indep.add_output('x1', shape_by_conn=True)

        par = p.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('G1', SeriesGroup(2,1, DistribDynShapeComp))
        par.add_subsystem('G2', SeriesGroup(2,1, DistribDynShapeComp))

        if solve_dyn_fwd:
            p.model.add_subsystem('sink', om.ExecComp(['y1=x1+x2'], x1={'shape_by_conn': True},
                                                      x2={'shape_by_conn': True},
                                                      y1={'copy_shape': 'x1'}))
        else:
            p.model.add_subsystem('sink', om.ExecComp(['y1=x1+x2'], shape=(8,)))

        p.model.connect('indep.x1', ['par.G1.C1.x1', 'par.G2.C1.x1'])
        p.model.connect('par.G1.C2.y1', 'sink.x1', src_indices=om.slicer[:])
        p.model.connect('par.G2.C2.y1', 'sink.x2', src_indices=om.slicer[:])
        return p

    def test_remote_distrib_err(self):
        # this test has remote distributed components (distributed comps under parallel groups)
        p = self._build_model(solve_dyn_fwd=False)

        with self.assertRaises(RuntimeError) as cm:
            p.setup()
            p.final_setup()

        print(cm.exception.args[0])

        self.assertTrue(
            "Collected errors for problem 'remote_distrib_err':\n"
            "   <model> <class Group>: Input 'sink.x1' has src_indices so the shape of connected output 'par.G1.C2.y1' cannot be determined.\n"
            "   <model> <class Group>: Input 'sink.x2' has src_indices so the shape of connected output 'par.G2.C2.y1' cannot be determined.\n"
            "   <model> <class Group>: Failed to resolve shapes for ['indep.x1', 'par.G1.C1.x1', 'par.G1.C1.y1', 'par.G1.C2.x1', 'par.G1.C2.y1', "
            "'par.G2.C1.x1', 'par.G2.C1.y1', 'par.G2.C2.x1', 'par.G2.C2.y1']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
           in cm.exception.args[0])


class TestDynShapeFeature(unittest.TestCase):
    def test_feature_fwd(self):

        p = om.Problem()
        p.model.add_subsystem('indeps', om.IndepVarComp('x', val=np.ones(5)))
        p.model.add_subsystem('comp', DynPartialsComp())
        p.model.add_subsystem('sink', om.ExecComp('y=x',
                                                  x={'shape_by_conn': True, 'copy_shape': 'y'},
                                                  y={'copy_shape': 'x'}))
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
                                                  y={'copy_shape': 'x'}))
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        J = p.compute_totals(of=['sink.y'], wrt=['comp.x'])
        assert_near_equal(J['sink.y', 'comp.x'], np.eye(5)*3.)


class DistCompDiffSizeKnownInput(om.ExplicitComponent):
    def setup(self):
        size = (self.comm.rank + 1) * 3
        self.add_input('x', val=np.random.random(size), distributed=True)
        self.add_output('y', val=np.zeros(size), distributed=True)

    def compute(self, inputs, outputs):
        outputs['y'] = inputs['x'] * 5


class DistCompUnknownInput(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True, distributed=True)
        self.add_output('y', copy_shape='x', distributed=True)

    def compute(self, inputs, outputs):
        outputs['y'] = inputs['x'] * 5


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDynShapeCombosNoSrcInds(unittest.TestCase):
    """
    This will test the dynamic shaping on parallel runs with all of the possible
    combinations of serial and distributed connections and dynamic shaping directions
    without src_indices.
    """

    N_PROCS = 3

    def test_serial_serial_fwd(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', val=np.random.random(2))
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                  x={'shape_by_conn': True},
                                                  y=np.zeros(2)))
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x')*2, p.get_val('comp.y'))

    def test_serial_serial_rev(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                  x=np.random.random(2),
                                                  y=np.zeros(2)))
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x')*2, p.get_val('comp.y'))

    def test_dist_dist_fwd(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        sizes = [3,0,5]
        indeps.add_output('x', np.random.random(sizes[p.comm.rank]), distributed=True)
        p.model.add_subsystem('comp', DistCompUnknownInput())
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x')*5, p.get_val('comp.y'))
        np.testing.assert_allclose(p.get_val('indeps.x', get_remote=True)*5, p.get_val('comp.y', get_remote=True))

    def test_dist_dist_rev(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True, distributed=True)
        p.model.add_subsystem('comp', DistCompDiffSizeKnownInput())
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        p.run_model()
        np.testing.assert_allclose(p.get_val('indeps.x')*5, p.get_val('comp.y'))
        np.testing.assert_allclose(p.get_val('indeps.x', get_remote=True)*5, p.get_val('comp.y', get_remote=True))


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDynShapeComboNoSrcIndsErrs(unittest.TestCase):

    N_PROCS = 3

    def test_serial_dist_rev_err(self):
        p = om.Problem(name='serial_dist_rev_err')
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', shape_by_conn=True)
        p.model.add_subsystem('comp', DistCompDiffSizeKnownInput())
        p.model.connect('indeps.x', 'comp.x')
        p.setup()
        with self.assertRaises(Exception) as cm:
            p.final_setup()

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'serial_dist_rev_err':"
           "\n   <model> <class Group>: dynamic sizing of non-distributed output 'indeps.x' from distributed input 'comp.x' is not supported because not all comp.x ranks are the same shape (shapes=[(3,), (6,), (9,)])."
           "\n   <model> <class Group>: Failed to resolve shapes for ['indeps.x']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
           "\n   'comp' <class DistCompDiffSizeKnownInput>: Can't determine src_indices automatically for input 'comp.x'. They must be supplied manually."
           "\n   <model> <class Group>: The source and target shapes do not match or are ambiguous for the connection 'indeps.x' to 'comp.x'. The source shape is (0,) but the target shape is (18,).")

    def test_dist_serial_fwd_err(self):
        p = om.Problem(name='dist_serial_fwd_err')
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', np.ones(3), distributed=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                  x={'shape_by_conn': True},
                                                  y={'copy_shape': 'x'}))
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
            p.final_setup()
        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'dist_serial_fwd_err':\n"
            "   <model> <class Group>: dynamic sizing of non-distributed input 'comp.x' from distributed output 'indeps.x' without src_indices is not supported.\n"
            "   <model> <class Group>: Failed to resolve shapes for ['comp.x', 'comp.y']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.\n"
            "   <model> <class Group>: Can't connect distributed output 'indeps.x' to non-distributed input 'comp.x' without specifying src_indices.")

    def test_dist_serial_rev_err(self):
        p = om.Problem(name='dist_serial_rev_err')
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', distributed=True, shape_by_conn=True)
        p.model.add_subsystem('comp', om.ExecComp('y = x * 2', shape=3))
        p.model.connect('indeps.x', 'comp.x')
        with self.assertRaises(Exception) as cm:
            p.setup()
            p.final_setup()
        self.assertTrue(
            "\nCollected errors for problem 'dist_serial_rev_err':"
            "\n   <model> <class Group>: Input 'comp.x' has src_indices so the shape of connected output 'indeps.x' cannot be determined."
            "\n   <model> <class Group>: Failed to resolve shapes for ['indeps.x']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
            "\n   <model> <class Group>: Can't connect distributed output 'indeps.x' to non-distributed input 'comp.x' without specifying src_indices." in cm.exception.args[0])


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
        self.add_output('y', compute_shape=lambda shapes: shapes['x'], units='ft')

    def compute(self, inputs, outputs):
        outputs['y'] = 3. * inputs['x']


class PGroup(om.Group):

    def setup(self):
        self.add_subsystem('comp1', DynShpComp(), promotes_inputs=['x'])
        self.add_subsystem('comp2', DynShpComp(), promotes_inputs=['x'])

    def configure(self):
        self.set_input_defaults('x', src_shape=(2, ))


class DanglingInputGroup(om.Group):

    def setup(self):
        self.add_subsystem('comp1', DynShpComp(), promotes_inputs=['x'])
        self.add_subsystem('comp2', DynShpComp(), promotes_inputs=['x'])


class TestDynShapesViaSetVal(unittest.TestCase):
    def test_group_dangling_input(self):
        prob = om.Problem()
        prob.model.add_subsystem('sub', DanglingInputGroup())

        prob.setup()

        # setting this sets the shape of sub.x
        prob['sub.x'] = np.ones(2) * 7.
        prob.run_model()

        assert_near_equal(prob['sub.comp1.y'], np.ones(2) * 21.)
        assert_near_equal(prob['sub.comp2.y'], np.ones(2) * 21.)


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
        sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'shape_by_conn': True}, y={'copy_shape': 'x'}),
                          promotes_inputs=['x'])
        sub.add_subsystem('comp2', om.ExecComp('y=3*x', x=np.ones(2), y=np.zeros(2)),
                          promotes_inputs=['x'])

        prob.setup()

        prob['sub.x'] = np.ones(2) * 7.
        prob.run_model()

        assert_near_equal(prob['sub.comp1.y'], np.ones(2) * 21.)
        assert_near_equal(prob['sub.comp2.y'], np.ones(2) * 21.)

    def test_shape_from_conn_input_mismatch(self):
        prob = om.Problem(name='shape_from_conn_input_mismatch')
        sub = prob.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'shape_by_conn': True}, y={'copy_shape': 'x'}),
                          promotes_inputs=['x'])
        sub.add_subsystem('comp2', om.ExecComp('y=3*x', x=np.ones(2), y=np.zeros(2)),
                          promotes_inputs=['x'])
        sub.add_subsystem('comp3', om.ExecComp('y=3*x', x=np.ones(3), y=np.zeros(3)),
                          promotes_inputs=['x'])

        with self.assertRaises(Exception) as cm:
            prob.setup()
            prob.final_setup()

        # just make sure we still get a clear error msg

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'shape_from_conn_input_mismatch':"
           "\n   <model> <class Group>: Shape of input 'sub.comp3.x', (3,), doesn't match shape (2,).")

    def test_shape_from_conn_input_mismatch_group_inputs(self):
        prob = om.Problem(name='shape_from_conn_input_mismatch_group_inputs')
        sub = prob.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'shape_by_conn': True}, y={'copy_shape': 'x'}),
                          promotes_inputs=['x'])
        sub.add_subsystem('comp2', om.ExecComp('y=3*x', x=np.ones(2), y=np.zeros(2)),
                          promotes_inputs=['x'])

        sub.set_input_defaults('x', src_shape=(3, ))

        with self.assertRaises(Exception) as cm:
            prob.setup()
            prob.final_setup()

        # just make sure we still get a clear error msg

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'shape_from_conn_input_mismatch_group_inputs':"
           "\n   <model> <class Group>: Shape of input 'sub.comp2.x', (2,), doesn't match shape (3,).")


class MatMatProd(om.ExplicitComponent):
    # matrix matrix product comp with computed output shape
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


        model.add_subsystem('C1', MatMatProd())
        model.add_subsystem('C2', MatMatProd())
        model.add_subsystem('C3', MatMatProd())

        model.connect('indep.M', 'C1.M')
        model.connect('indep.N', 'C1.N')
        model.connect('indep.O', 'C2.M')
        model.connect('indep.P', 'C2.N')
        model.connect('C1.out', 'C3.M')
        model.connect('C2.out', 'C3.N')

        p.setup()
        p.run_model()

        self.assertEqual(model.C3._outputs['out'].shape, (3, 7))


@unittest.skipIf(PETScVector is None, 'test requires PETSc')
class TestDynShapeSrcIndices(unittest.TestCase):
    N_PROCS = 2

    def test_serial_serial_fwd(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', val=np.ones((2, 3)))

        model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                x={'shape_by_conn': True}, y={'copy_shape': 'x'}))

        model.connect('indep.x', 'comp.x')

        p.setup()
        p.run_model()
        assert_near_equal(p.get_val('comp.y'), np.ones((2, 3)) * 2)

    def test_serial_serial_fwd_src_inds(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', val=np.ones((2, 3)))

        model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                x={'shape_by_conn': True}, y={'copy_shape': 'x'}))

        model.connect('indep.x', 'comp.x', src_indices=om.slicer[:, [0,2]], flat_src_indices=False)

        p.setup()
        p.run_model()
        assert_near_equal(p.get_val('comp.y'), np.ones((2, 2)) * 2)

    def test_serial_dist_fwd(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', val=np.ones((2, 3)))

        model.add_subsystem('comp', DistribDynShapeComp(n_inputs=1))

        model.connect('indep.x', 'comp.x1')

        p.setup()
        p.run_model()
        assert_near_equal(p.get_val('comp.y1', get_remote=False), np.ones((2, 3)) * 2)

        # serial --> dist results in dist value in each proc being equivalent to the serial value,
        # so the full dist value across 2 procs is (4, 3)
        assert_near_equal(p.get_val('comp.y1', get_remote=True), np.ones((4, 3)) * 2)

    def test_dist_serial_fwd_flat(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', val=np.ones((2, 3)), distributed=True)

        model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                x={'shape_by_conn': True}, y={'copy_shape': 'x'}))

        model.connect('indep.x', 'comp.x', src_indices=om.slicer[:-3], flat_src_indices=True)

        p.setup()
        p.run_model()
        assert_near_equal(p.get_val('comp.y'), np.ones((9,)) * 2)

    def test_dist_serial_fwd_nonflat(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', val=np.ones((2, 3)), distributed=True)

        model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                x={'shape_by_conn': True}, y={'copy_shape': 'x'}))

        model.connect('indep.x', 'comp.x', src_indices=om.slicer[:, [0,2]], flat_src_indices=False)

        p.setup()
        p.run_model()
        assert_near_equal(p.get_val('comp.y'), np.ones((4,2)) * 2)

    def test_dist_dist_fwd_nonflat(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        val = np.array([[2, 3, 2, 3],
                        [2, 3, 2, 3]], dtype=float)
        indep.add_output('x', val=val, distributed=True)

        model.add_subsystem('comp', DistribDynShapeComp(n_inputs=1))  # comp multiplies by 2

        if p.comm.rank == 0:
            model.connect('indep.x', 'comp.x1', src_indices=om.slicer[:, [0,2]], flat_src_indices=False)
        else:
            model.connect('indep.x', 'comp.x1', src_indices=om.slicer[:, [1,3]], flat_src_indices=False)

        p.setup()
        p.run_model()
        if p.comm.rank == 0:
            assert_near_equal(p.get_val('comp.y1', get_remote=False), np.ones((4, 2)) * 4)
        else:
            assert_near_equal(p.get_val('comp.y1', get_remote=False), np.ones((4, 2)) * 6)

        global_val = np.ones((8, 2))
        global_val[:4] *= 4
        global_val[4:] *= 6
        assert_near_equal(p.get_val('comp.y1', get_remote=True), global_val)

    def test_dist_dist_fwd_flat(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        if p.comm.rank == 0:
            val = np.array([2,2,2,2], dtype=float)
        else:
            val = np.array([3,3,3,3], dtype=float)
        indep.add_output('x', val=val, distributed=True)

        model.add_subsystem('comp', DistribDynShapeComp(n_inputs=1))  # comp multiplies by 2

        if p.comm.rank == 0:
            model.connect('indep.x', 'comp.x1', src_indices=om.slicer[4:], flat_src_indices=True)
        else:
            model.connect('indep.x', 'comp.x1', src_indices=om.slicer[:4], flat_src_indices=True)

        p.setup()
        p.run_model()
        if p.comm.rank == 0:
            assert_near_equal(p.get_val('comp.y1', get_remote=False), np.ones(4) * 6)
        else:
            assert_near_equal(p.get_val('comp.y1', get_remote=False), np.ones(4) * 4)

        global_val = np.ones((8,))
        global_val[:4] *= 6
        global_val[4:] *= 4
        assert_near_equal(p.get_val('comp.y1', get_remote=True), global_val)

    def test_serial_serial_rev(self):
        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', shape_by_conn=True)

        model.add_subsystem('comp', om.ExecComp('y = x * 2',
                                                x={'copy_shape': 'y'}, y={'shape_by_conn': True}))
        model.add_subsystem('sink', om.ExecComp('y = x', shape=(2,3)))

        model.connect('indep.x', 'comp.x')
        model.connect('comp.y', 'sink.x')

        p.setup()
        p.run_model()
        assert_near_equal(p.get_val('indep.x'), np.ones((2, 3)))

    def test_serial_dist_rev(self):

        class MyDistComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', distributed=True, shape=(2,3))
                self.add_output('y', distributed=True, shape=(2,3))

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] * 2.

        p = om.Problem()
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', shape_by_conn=True)

        model.add_subsystem('comp', MyDistComp())

        model.connect('indep.x', 'comp.x')

        p.setup()
        p.run_model()
        assert_near_equal(p.get_val('indep.x'), np.ones((2, 3)))

    def test_dist_serial_rev_err(self):

        p = om.Problem(name='dist_serial')
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', shape_by_conn=True, distributed=True)

        model.add_subsystem('comp', om.ExecComp('y=2*x', shape=(2,3)))

        model.connect('indep.x', 'comp.x', src_indices=om.slicer[:], flat_src_indices=False)

        p.setup()

        with self.assertRaises(Exception) as cm:
            p.final_setup()

        # just make sure we still get a clear error msg

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'dist_serial':"
           "\n   <model> <class Group>: Input 'comp.x' has src_indices so the shape of connected output 'indep.x' cannot be determined."
           "\n   <model> <class Group>: Failed to resolve shapes for ['indep.x']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'.")

    def test_dist_dist_rev_err(self):
        class MyDistComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', distributed=True, shape=(2,3))
                self.add_output('y', distributed=True, shape=(2,3))

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] * 2.

        p = om.Problem(name='dist_dist')
        model = p.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', shape_by_conn=True, distributed=True)

        model.add_subsystem('comp', MyDistComp())
        model.connect('indep.x', 'comp.x', src_indices=om.slicer[:, [0,2]])

        p.setup()

        with self.assertRaises(Exception) as cm:
            p.final_setup()

        # just make sure we still get a clear error msg

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'dist_dist':"
           "\n   <model> <class Group>: Input 'comp.x' has src_indices so the shape of connected output 'indep.x' cannot be determined."
           "\n   <model> <class Group>: Failed to resolve shapes for ['indep.x']. To see the dynamic shapes dependency graph, do 'openmdao view_dyn_shapes <your_py_file>'."
           "\n   <model> <class Group>: When connecting 'indep.x' to 'comp.x': Can't set source shape to (0,) because indexer (slice(None, None, None), [0, 2]) expects 2 dimensions.")


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
