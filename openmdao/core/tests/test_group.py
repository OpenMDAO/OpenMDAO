import unittest

from openmdao.api import Problem, Group, ExecComp

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
