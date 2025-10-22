import unittest
import openmdao.api as om
import numpy as np
from openmdao.visualization.conn_graph import is_equal, are_compatible_values


class TestAllConnectionTypes(unittest.TestCase):

    def build_nested_model(self, promote=False, autoivc=False):
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

    def test_promoted(self):
        prob = self.build_nested_model(promote=True)
        model = prob.model

        model.connect('G1.G1A.y', 'G2.G2A.a')
        model.connect('G2.G2A.b', 'G2.G2B.b')

        model.connect('G1.G1B.w', 'C7.w')
        model.connect('G2.G2B.d', 'C7.d')

        prob.setup()
        prob.run_model()

        # prob.model._get_all_conn_graph().display()
        # prob.model._get_all_conn_graph().dump()
        #print('done')

    def test_promoted_branching_output(self):
        prob = self.build_nested_model(promote=True)
        model = prob.model

        # model.connect('G1.G1A.y', 'G2.G2A.a')
        model.connect('x', 'G2.G2A.a')
        model.connect('G2.G2A.b', 'G2.G2B.b')

        model.connect('G1.G1B.w', 'C7.w')
        model.connect('G2.G2B.d', 'C7.d')

        prob.setup()
        prob.run_model()

        #prob.model.display_conn_graph()
        #prob.model.G2.display_conn_graph()
        #prob.model._get_all_conn_graph().dump()
        #print('done')

    def test_promoted_branching_output_autoivc(self):
        prob = self.build_nested_model(promote=True, autoivc=True)
        model = prob.model

        # model.connect('G1.G1A.y', 'G2.G2A.a')
        #model.connect('x', 'G2.G2A.a')
        model.connect('G2.G2A.a', 'x')
        model.connect('G2.G2A.b', 'G2.G2B.b')

        model.connect('G1.G1B.w', 'C7.w')
        model.connect('G2.G2B.d', 'C7.d')

        prob.setup()
        prob.run_model()

        #prob.model._get_all_conn_graph().display()
        #prob.model._get_all_conn_graph().dump()
        #print('done')

    def test_promoted_autoivc(self):
        prob = self.build_nested_model(promote=True)

        prob.setup()
        prob.run_model()

        # prob.model._get_all_conn_graph().display()

    def test_no_promotion(self):
        prob = self.build_nested_model(promote=False)
        model = prob.model

        model.connect('G1.G1A.C1.y', 'G2.G2A.C4.a')
        model.connect('G2.G2A.C4.b', 'G2.G2B.C5.b')

        model.connect('G1.G1B.C3.w', 'C7.w')
        model.connect('G2.G2B.C6.d', 'C7.d')

        prob.setup()
        prob.run_model()

        #prob.model._get_all_conn_graph().display()

    def test_no_promotion_input_to_input(self):
        prob = self.build_nested_model(promote=False)
        model = prob.model

        model.connect('G1.G1A.C1.x', 'G2.G2A.C4.a')  # input-input
        model.connect('G2.G2A.C4.b', 'G2.G2B.C5.b')

        model.connect('G1.G1B.C3.w', 'C7.w')
        model.connect('G2.G2B.C6.d', 'C7.d')

        prob.setup()
        prob.run_model()

        #prob.model.display_conn_graph()
        #prob.model._get_all_conn_graph().dump()
        #print('done')


class TestAllConnGraphUtilityFunctions(unittest.TestCase):
    """Test utility functions in AllConnGraph."""

    def test_is_equal(self):
        """Test is_equal function."""
        # Test with equal arrays
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        self.assertTrue(is_equal(a, b))

        # Test with different arrays
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        self.assertFalse(is_equal(a, b))

        # Test with scalars
        self.assertTrue(is_equal(5, 5))
        self.assertFalse(is_equal(5, 6))

        # Test with different types
        self.assertFalse(is_equal(5, 5.0))

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



if __name__ == '__main__':
    unittest.main()
