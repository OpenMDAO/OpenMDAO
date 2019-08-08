import unittest
import openmdao.api as om
from openmdao.core.tests.test_impl_comp import QuadraticComp
from six.moves import cStringIO

class TestRootComponentListOutputs(unittest.TestCase):
    """ Testing list_outputs for Component as root model."""

    def test_list_outputs(self):
        """Test list_outputs when the root model is a Component."""
        group = om.Group()

        comp1 = group.add_subsystem('comp1', om.IndepVarComp())
        comp1.add_output('a', 1.0)
        comp1.add_output('b', 1.0)
        comp1.add_output('c', 1.0)

        sub = group.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp2', QuadraticComp())
        sub.add_subsystem('comp3', QuadraticComp())

        group.connect('comp1.a', 'sub.comp2.a')
        group.connect('comp1.b', 'sub.comp2.b')
        group.connect('comp1.c', 'sub.comp2.c')

        group.connect('comp1.a', 'sub.comp3.a')
        group.connect('comp1.b', 'sub.comp3.b')
        group.connect('comp1.c', 'sub.comp3.c')

        prob = om.Problem(model=group)
        prob.setup()

        prob['comp1.a'] = 1.
        prob['comp1.b'] = -4.
        prob['comp1.c'] = 3.
        prob.run_model()

        prob2 = om.Problem(model=comp1)
        stream2 = cStringIO()
        outputs2 = prob2.model.list_outputs(hierarchical=False, out_stream=stream2)
        self.assertEqual(sorted(outputs2), [
            ('comp1.a', {'value': [1.]}),
            ('comp1.b', {'value': [-4.]}),
            ('comp1.c', {'value': [3.]})
        ])
        text2 = stream2.getvalue()
        self.assertEqual(text2.count('comp1.'), 3)

        prob3 = om.Problem(model=sub.comp2)
        stream3 = cStringIO()
        outputs3 = prob3.model.list_outputs(hierarchical=False, out_stream=stream3)
        self.assertEqual(outputs3, [
            ('sub.comp2.x', {'value': [3.]})
        ])
        text3 = stream3.getvalue()
        self.assertEqual(text3.count('sub.comp2.x'), 1)

if __name__ == '__main__':
    unittest.main()