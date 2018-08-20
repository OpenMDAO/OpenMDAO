import unittest

from openmdao.api import Problem, IndepVarComp

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None


class TestVector(unittest.TestCase):

    def test_keys(self):
        p = Problem()
        comp = IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()
        p.final_setup()

        keys = sorted(p.model._outputs.keys())
        expected = ['des_vars.v1', 'des_vars.v2']

        self.assertListEqual(keys, expected, msg='keys() is not returning the expected names')

    def test_iter(self):
        p = Problem()
        comp = IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()
        p.final_setup()

        outputs = [n for n in p.model._outputs]
        expected = ['des_vars.v1', 'des_vars.v2']

        self.assertListEqual(outputs, expected, msg='Iter is not returning the expected names')

    def test_dot(self):
        p = Problem()
        comp = IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()
        p.final_setup()

        new_vec = p.model._outputs._clone()
        new_vec.set_const(3.)

        self.assertEqual(new_vec.dot(p.model._outputs), 9.)

    def test_dot_petsc(self):
        if not PETScVector:
            raise unittest.SkipTest("PETSc is not installed")

        p = Problem()
        comp = IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()
        p.final_setup()

        new_vec = p.model._outputs._clone()
        new_vec.set_const(3.)

        self.assertEqual(new_vec.dot(p.model._outputs), 9.)


if __name__ == '__main__':
    unittest.main()
