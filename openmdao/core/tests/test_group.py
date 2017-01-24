import unittest

import numpy as np

from openmdao.api import Problem, Group, ExecComp, IndepVarComp

class TestGroup(unittest.TestCase):

    def test_same_sys_name(self):
        p = Problem(model=Group())
        p.model.add_subsystem("foo", ExecComp("y=2.0*x"))
        p.model.add_subsystem("bar", ExecComp("y=2.0*x"))

        try:
            p.model.add_subsystem("foo", ExecComp("y=2.0*x"))
        except Exception as err:
            self.assertEqual(str(err), "Subsystem name 'foo' is already used.")
        else:
            self.fail("Exception expected.")


class TestConnect(unittest.TestCase):

    def setUp(self):
        prob = Problem(Group())

        prob.model.add_subsystem('src', IndepVarComp('x', np.zeros(5,)))
        prob.model.add_subsystem('tgt', ExecComp('y = x'))
        prob.model.add_subsystem('tgt2', ExecComp('y = x'))

        prob.setup(check=False)

        self.model = prob.model

    def test_invalid_target(self):
        with self.assertRaises(NameError):
            self.model.connect('src.x', 'tgt.z', src_indices=[1])

    def test_src_indices_as_int_list(self):
        self.model.connect('src.x', 'tgt.x', src_indices=[1])

    def test_src_indices_as_int_array(self):
        self.model.connect('src.x', 'tgt.x', src_indices=np.zeros(1, dtype=int))

    def test_src_indices_as_float_list(self):
        with self.assertRaises(TypeError):
            self.model.connect('src.x', 'tgt.x', src_indices=[1.0])

    def test_src_indices_as_float_array(self):
        with self.assertRaises(TypeError):
            self.model.connect('src.x', 'tgt.x', src_indices=np.zeros(1))

    def test_src_indices_as_str(self):
        with self.assertRaises(TypeError):
            self.model.connect('src.x', 'tgt.x', 'tgt2.x')


if __name__ == "__main__":
    unittest.main()
