from mpi4py import MPI
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
        # Inputs
        self.add_input('in', copy_shape='out')
        self.add_output('out', shape_by_conn=True)

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class C(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('in', shape=4)
        self.add_output('out', shape=9)

    def compute(self, inputs, outputs):
        outputs['out'] = np.arange(9)


class D(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class E(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        print(inputs['in'])
        outputs['out'] = inputs['in']


class B_distrib(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        # Inputs
        self.add_input('in', copy_shape='out')
        self.add_output('out', shape_by_conn=True)

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class C_distrib(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        # Inputs
        if self.comm.rank == 0:
            self.add_input('in', shape=1, src_indices=np.arange(0,1, dtype=int))
        elif self.comm.rank == 1:
            self.add_input('in', shape=2, src_indices=np.arange(1,3, dtype=int))
        else:
            self.add_input('in', shape=0, src_indices=np.arange(3,3, dtype=int))

        self.add_output('out', shape=3)

    def compute(self, inputs, outputs):
        outputs['out'] *= self.comm.rank


class D_distrib(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        # Inputs
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

        msg = "Group (<model>): Failed to resolve shapes for ['B.in', 'B.out', 'C.in', 'C.out']."
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
        self.assertEqual(np.sum(prob.get_val('E.out')), (n**2 + n)/2 * size_down)


class ResizableComp(om.ExplicitComponent):
    # this is just a component that allows us to resize between setups
    def __init__(self, n_inputs=1, size=5, mult=2.):
        super(ResizableComp, self).__init__()
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
        super(DynShapeComp, self).__init__()
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
        super(DistribDynShapeComp, self).__init__()
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
        super(DistribComp, self).__init__()
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
        super(DynShapeGroupSeries, self).__init__()
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
        super(DynShapeGroupConnectedInputs, self).__init__()
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

        msg = "Group (<model>): Failed to resolve shapes for ['Gdyn.C1.x2', 'Gdyn.C1.y2', 'Gdyn.C2.x2', 'Gdyn.C2.y2', 'Gdyn.C3.x2', 'Gdyn.C3.y2', 'sink.x2', 'sink.y2']."
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


if __name__ == "__main__":
    unittest.main()
