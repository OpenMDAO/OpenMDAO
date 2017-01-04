import unittest

from openmdao.api import Problem, Group, ExecComp

class TestGroup(unittest.TestCase):

    def test_same_sys_name(self):
        p = Problem(root=Group())
        p.root.add_subsystem("foo", ExecComp("y=2.0*x"))
        p.root.add_subsystem("bar", ExecComp("y=2.0*x"))

        try:
            p.root.add_subsystem("foo", ExecComp("y=2.0*x"))
        except Exception as err:
            self.assertEqual(str(err), "Subsystem name 'foo' is already used.")
        else:
            self.fail("Exception expected.")
