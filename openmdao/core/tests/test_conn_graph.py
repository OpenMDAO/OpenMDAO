import unittest
import openmdao.api as om
import numpy as np
from openmdao.core.conn_graph import are_compatible_values
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_near_equal


try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


def build_nested_model(promote=False, autoivc=False):
    prob = om.Problem()
    model = prob.model

    if promote:
        def get_proms(lst):
            return lst
    else:
        def get_proms(lst):
            return None

    if not autoivc:
        model.add_subsystem('indeps', om.IndepVarComp('x', 10.0), promotes=get_proms(['x']))

    G1 = model.add_subsystem('G1', om.Group(), promotes_inputs=get_proms(['x']))
    G2 = model.add_subsystem('G2', om.Group())

    G1A = G1.add_subsystem('G1A', om.Group(), promotes_inputs=get_proms(['x']))
    G1B = G1.add_subsystem('G1B', om.Group(), promotes_inputs=get_proms([('y','x')]))
    G2A = G2.add_subsystem('G2A', om.Group())
    G2B = G2.add_subsystem('G2B', om.Group())

    G1A.add_subsystem('C1', om.ExecComp('y = 2.0 * x'), promotes=get_proms(['x', 'y']))

    G1B.add_subsystem('C2', om.ExecComp('z = 3.0 * y'), promotes=get_proms(['z', 'y']))
    G1B.add_subsystem('C3', om.ExecComp('w = 4.0 * z'), promotes=get_proms(['w', 'z']))

    G2A.add_subsystem('C4', om.ExecComp('b = 2.0 * a'), promotes=get_proms(['a', 'b']))

    G2B.add_subsystem('C5', om.ExecComp('c = 3.0 * b'), promotes=get_proms(['b', 'c']))
    G2B.add_subsystem('C6', om.ExecComp('d = 4.0 * c'), promotes=get_proms(['c', 'd']))

    model.add_subsystem('C7', om.ExecComp('result = w + d'))

    return prob


class TestAllConnectionTypes(unittest.TestCase):

    def test_promoted(self):
        prob = build_nested_model(promote=True)
        model = prob.model

        model.connect('G1.G1A.y', 'G2.G2A.a')
        model.connect('G2.G2A.b', 'G2.G2B.b')

        model.connect('G1.G1B.w', 'C7.w')
        model.connect('G2.G2B.d', 'C7.d')

        prob.setup()
        prob.run_model()

    def test_promoted_branching_output(self):
        prob = build_nested_model(promote=True)
        model = prob.model

        model.connect('x', 'G2.G2A.a')
        model.connect('G2.G2A.b', 'G2.G2B.b')

        model.connect('G1.G1B.w', 'C7.w')
        model.connect('G2.G2B.d', 'C7.d')

        prob.setup()
        prob.run_model()

    def test_promoted_branching_output_autoivc(self):
        prob = build_nested_model(promote=True, autoivc=True)
        model = prob.model

        model.connect('G2.G2A.a', 'x')
        model.connect('G2.G2A.b', 'G2.G2B.b')

        model.connect('G1.G1B.w', 'C7.w')
        model.connect('G2.G2B.d', 'C7.d')

        prob.setup()
        prob.run_model()

    def test_promoted_autoivc(self):
        prob = build_nested_model(promote=True)

        prob.setup()
        prob.run_model()

    def test_no_promotion(self):
        prob = build_nested_model(promote=False)
        model = prob.model

        model.connect('G1.G1A.C1.y', 'G2.G2A.C4.a')
        model.connect('G2.G2A.C4.b', 'G2.G2B.C5.b')

        model.connect('G1.G1B.C3.w', 'C7.w')
        model.connect('G2.G2B.C6.d', 'C7.d')

        prob.setup()
        prob.run_model()


class TestInputToInputConnections(unittest.TestCase):

    def test_no_promotion(self):
        prob = build_nested_model(promote=False, autoivc=True)
        model = prob.model

        model.connect('G1.G1A.C1.x', 'G2.G2A.C4.a')  # input-input
        model.connect('G2.G2A.C4.b', 'G2.G2B.C5.b')

        model.connect('G1.G1B.C3.w', 'C7.w')
        model.connect('G2.G2B.C6.d', 'C7.d')

        prob.setup()
        prob.run_model()

    def test_promotion_autoivc(self):
        prob = build_nested_model(promote=True, autoivc=True)
        model = prob.model

        model.connect('x', 'G2.G2B.b')  # input-input

        prob.setup()
        prob.final_setup()
        prob.run_model()

        # set outputs=False since name of auto_ivc may change between runs
        conn_tree = prob.model.get_conn_graph().get_conn_tree_graph(('i', 'x'), outputs=False)
        expected =  set([(('i', 'x'), ('i', 'G1.x')),
                        (('i', 'G1.x'), ('i', 'G1.G1A.x')),
                        (('i', 'G1.G1A.x'), ('i', 'G1.G1A.C1.x')),
                        (('i', 'G1.x'), ('i', 'G1.G1B.y')),
                        (('i', 'G1.G1B.y'), ('i', 'G1.G1B.C2.y')),
                        (('i', 'G2.G2B.b'), ('i', 'G2.G2B.C5.b')),
                        ])
        self.assertEqual(set(conn_tree.edges()), expected)

    def test_promotion(self):
        prob = build_nested_model(promote=True, autoivc=False)
        model = prob.model

        model.connect('x', 'G2.G2B.b')  # input-input

        prob.setup()
        prob.final_setup()
        prob.run_model()

    def test_src_indices(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indeps', om.IndepVarComp())
        indep.add_output('x', val=np.arange(10) + 1.0)

        model.add_subsystem('C1', om.ExecComp('y = 2.0 * x', x=np.zeros(10), y=np.zeros(10)))
        model.add_subsystem('C2', om.ExecComp('y = 3.0 * x', x=np.zeros(5), y=np.zeros(5)))

        model.connect('indeps.x', 'C1.x')
        model.connect('C1.x', 'C2.x', src_indices=[0, 2, 4, 6, 8])  # input-input with src_indices

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('C1.x'), np.arange(10) + 1.0)
        assert_near_equal(prob.get_val('C1.y'), np.array([2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]))
        assert_near_equal(prob.get_val('C2.x'), np.array([1., 3., 5., 7., 9.]))
        assert_near_equal(prob.get_val('C2.y'), np.array([3., 9., 15., 21., 27.]))

    def test_src_indices_2_levels(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indeps', om.IndepVarComp())
        indep.add_output('x', val=np.arange(15) + 1.0)

        model.add_subsystem('C1', om.ExecComp('y = 2.0 * x', x=np.zeros(10), y=np.zeros(10)))
        model.add_subsystem('C2', om.ExecComp('y = 3.0 * x', x=np.zeros(5), y=np.zeros(5)))

        model.connect('indeps.x', 'C1.x', src_indices=om.slicer[:10])
        model.connect('C1.x', 'C2.x', src_indices=[0, 2, 4, 6, 8])  # input-input with src_indices

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('C1.x'), np.arange(10) + 1.0)
        assert_near_equal(prob.get_val('C1.y'), np.array([2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]))

        assert_near_equal(prob.get_val('C2.x'), np.array([1., 3., 5., 7., 9.]))
        assert_near_equal(prob.get_val('C2.y'), np.array([3., 9., 15., 21., 27.]))

    def test_src_indices_2_levels_autoivc(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', om.ExecComp('y = 2.0 * x', x=np.arange(15) + 1.0, y=np.zeros(15)))
        model.add_subsystem('C2', om.ExecComp('y = 3.0 * x', x=np.zeros(10), y=np.zeros(10)))
        model.add_subsystem('C3', om.ExecComp('y = 1.5 * x', x=np.zeros(5), y=np.zeros(5)))

        model.connect('C1.x', 'C2.x', src_indices=om.slicer[:10])  # input-input with src_indices
        model.connect('C2.x', 'C3.x', src_indices=[0, 2, 4, 6, 8])  # input-input with src_indices

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('C1.x'), np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]))
        assert_near_equal(prob.get_val('C1.y'), np.array([2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30.]))
        assert_near_equal(prob.get_val('C2.x'), np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]))
        assert_near_equal(prob.get_val('C2.y'), np.array([3., 6., 9., 12., 15., 18., 21., 24., 27., 30.]))
        assert_near_equal(prob.get_val('C3.x'), np.array([1., 3., 5., 7., 9.]))
        assert_near_equal(prob.get_val('C3.y'), np.array([1.5, 4.5, 7.5, 10.5, 13.5]))


class TestAllConnGraphUtilityFunctions(unittest.TestCase):
    """Test utility functions in AllConnGraph."""

    def test_are_compatible_values(self):
        """Test are_compatible_values function."""
        # Test with compatible arrays
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        self.assertTrue(are_compatible_values(a, b, False))

        # Test with incompatible shapes
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse(are_compatible_values(a, b, False))

        # Test with different types
        self.assertFalse(are_compatible_values(5, 5.0, True))


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestMPI(unittest.TestCase):

    N_PROCS = 2

    def test_parallel_group(self):
        p = om.Problem()
        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('x', 10.0), promotes=['x'])
        par = model.add_subsystem('par', om.ParallelGroup(), promotes=['x', 'y', 'z'])
        par.add_subsystem('C1', om.ExecComp('y = 2.0 * x'), promotes=['y', 'x'])
        par.add_subsystem('C2', om.ExecComp('z = 3.0 * x'), promotes=['z', 'x'])
        model.add_subsystem('C3', om.ExecComp('w = 4.0 * y + 5.0 * z'), promotes=['y', 'z'])
        p.setup()
        p.run_model()


if __name__ == '__main__':
    unittest.main()
