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
        prob.model.add_subsystem('cmp', ExecComp('z = x'))

        self.prob = prob
        self.model = prob.model

    def test_src_indices_as_int_list(self):
        self.model.connect('src.x', 'tgt.x', src_indices=[1])

    def test_src_indices_as_int_array(self):
        self.model.connect('src.x', 'tgt.x', src_indices=np.zeros(1, dtype=int))

    def test_src_indices_as_float_list(self):
        msg = "src_indices must contain integers, but src_indices for " + \
              "connection from src.x to tgt.x contains non-integers."

        with self.assertRaisesRegexp(TypeError, msg):
            self.model.connect('src.x', 'tgt.x', src_indices=[1.0])

    def test_src_indices_as_float_array(self):
        msg = "src_indices must contain integers, but src_indices for " + \
              "connection from src.x to tgt.x is <class 'numpy.float64'>."

        with self.assertRaisesRegexp(TypeError, msg):
            self.model.connect('src.x', 'tgt.x', src_indices=np.zeros(1))

    def test_src_indices_as_str(self):
        msg = "src_indices must be an index array, " + \
              "did you mean connect('src.x', [tgt.x, cmp.x])?"

        with self.assertRaisesRegexp(TypeError, msg):
            self.model.connect('src.x', 'tgt.x', 'cmp.x')

    def test_already_connected(self):
        msg = "Input 'tgt.x' is already connected to 'src.x'."

        self.model.connect('src.x', 'tgt.x')
        with self.assertRaisesRegexp(RuntimeError, msg):
            self.model.connect('cmp.x', 'tgt.x')

    def test_invalid_source(self):
        msg = "Output 'src.z' does not exist for connection " + \
              "in model from src.z to tgt.x."

        # source and target names can't be checked until setup
        # because initialize_variables is not called until then
        self.model.connect('src.z', 'tgt.x')
        with self.assertRaisesRegexp(NameError, msg):
            self.prob.setup(check=False)

    def test_invalid_target(self):
        msg = "Input 'tgt.z' does not exist for connection " + \
              "in model from src.x to tgt.z."

        # source and target names can't be checked until setup
        # because initialize_variables is not called until then
        self.model.connect('src.x', 'tgt.z')
        with self.assertRaisesRegexp(NameError, msg):
            self.prob.setup(check=False)


if __name__ == "__main__":
    unittest.main()
