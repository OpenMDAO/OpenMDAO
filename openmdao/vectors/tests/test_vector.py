import unittest

from openmdao.api import Problem, IndepVarComp

class TestVector(unittest.TestCase):

    def test_iter(self):

        p = Problem()
        comp = IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()

        outputs = [n for n in p.model._outputs]
        expected = ['des_vars.v1', 'des_vars.v2']

        self.assertListEqual(outputs, expected, msg='Iter is not returning the expected names')

if __name__ == '__main__':

    unittest.main()

