import unittest
from io import StringIO
import openmdao.api as om

class TestRootComponentListOutputs(unittest.TestCase):
    """ Testing list_outputs for Component as root model."""

    def test_list_outputs(self):
        """Test list_outputs when the root model is a Component."""

        comp1 = om.IndepVarComp()
        comp1.add_output('a', 1.0)
        comp1.add_output('b', 2.0)
        comp1.add_output('c', 3.0)

        prob = om.Problem()
        prob.model.add_subsystem('comp1', comp1)
        prob.setup()
        prob.run_model()

        stream = StringIO()
        outputs = prob.model.list_outputs(hierarchical=False, out_stream=stream)

        self.assertEqual(sorted(outputs), [
            ('comp1.a', {'val': [1.]}),
            ('comp1.b', {'val': [2.]}),
            ('comp1.c', {'val': [3.]})
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp1.'), 3)

if __name__ == '__main__':
    unittest.main()
