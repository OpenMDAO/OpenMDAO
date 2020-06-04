import unittest
from io import StringIO
import openmdao.api as om

class TestRootComponentListOutputs(unittest.TestCase):
    """ Testing list_outputs for Component as root model."""

    def test_list_outputs(self):
        """Test list_outputs when the root model is a Component."""

        comp1 = om.IndepVarComp()
        comp1.add_output('comp1_a', 1.0)
        comp1.add_output('comp1_b', 2.0)
        comp1.add_output('comp1_c', 3.0)

        prob = om.Problem(model=comp1)
        prob.setup()
        prob.run_model()

        stream = StringIO()
        outputs = prob.model.list_outputs(hierarchical=False, out_stream=stream)

        self.assertEqual(sorted(outputs), [
            ('comp1_a', {'value': [1.]}),
            ('comp1_b', {'value': [2.]}),
            ('comp1_c', {'value': [3.]})
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp1_'), 3)

if __name__ == '__main__':
    unittest.main()
