import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.devtools.testutil import TestLogger
from openmdao.error_checking.check_config import get_cycles

class MyComp(ExecComp):
    def __init__(self):
        super(MyComp, self).__init__(["y = 2.0*a", "z = 3.0*b"])

class TestCheckConfig(unittest.TestCase):

    def test_hanging_inputs(self):
        p = Problem(root=Group())
        root = p.root

        G1 = root.add_subsystem("G1", Group(), promotes=['*'])
        G2 = G1.add_subsystem("G2", Group(), promotes=['*'])
        C1 = G2.add_subsystem("C1", ExecComp('y=x*2.0+w'), promotes=['*'])
        C2 = G2.add_subsystem("C2", IndepVarComp('x', 1.0), promotes=['*'])

        G3 = root.add_subsystem("G3", Group())
        G4 = G3.add_subsystem("G4", Group())
        C3 = G4.add_subsystem("C3", ExecComp('y=x*2.0+u'), promotes=['*'])
        C4 = G4.add_subsystem("C4", ExecComp('y=x*2.0+v'))

        testlogger = TestLogger()
        p.setup(logger=testlogger)

        self.assertEqual(len(testlogger.get('warning')), 1)

        expected = [
            'G1.G2.C1.w',
            'G3.G4.C3.u',
            'G3.G4.C3.x',
            'G3.G4.C4.v',
            'G3.G4.C4.x'
        ]

        actual = testlogger.get('warning')[0].split('[', 1)[1].split(']')[0].split(',')
        actual = [a.strip().strip("'") for a in actual]

        self.assertEqual(expected, actual)

    def test_cycles_1_level(self):

        p = Problem(root=Group())
        root = p.root

        indep = root.add_subsystem("indep", IndepVarComp('x', 1.0))
        C1 = root.add_subsystem("C1", MyComp())
        C2 = root.add_subsystem("C2", MyComp())
        C3 = root.add_subsystem("C3", MyComp())
        C4 = root.add_subsystem("C4", MyComp())

        root.connect("C4.y", "C2.a")
        root.connect("C4.y", "C3.a")
        root.connect("C2.y", "C1.a")
        root.connect("C1.y", "C4.a")

        # make sure no system has dangling inputs so we avoid that warning
        root.connect("indep.x", "C1.b")
        root.connect("indep.x", "C2.b")
        root.connect("indep.x", "C3.b")
        root.connect("indep.x", "C4.b")

        testlogger = TestLogger()
        p.setup(logger=testlogger)

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 1)

        self.assertEqual(warnings[0] ,"Group '' has the following cycles: [['C1', 'C2', 'C4']]")

    def test_cycles_multi_level(self):

        p = Problem(root=Group())
        root = p.root

        indep = root.add_subsystem("indep", IndepVarComp('x', 1.0))

        G1 = root.add_subsystem("G1", Group())

        C1 = G1.add_subsystem("C1", MyComp())
        C2 = G1.add_subsystem("C2", MyComp())

        C3 = root.add_subsystem("C3", MyComp())
        C4 = root.add_subsystem("C4", MyComp())

        root.connect("C4.y", "G1.C2.a")
        root.connect("C4.y", "C3.a")
        root.connect("G1.C2.y", "G1.C1.a")
        root.connect("G1.C1.y", "C4.a")

        # make sure no system has dangling inputs so we avoid that warning
        root.connect("indep.x", "G1.C1.b")
        root.connect("indep.x", "G1.C2.b")
        root.connect("indep.x", "C3.b")
        root.connect("indep.x", "C4.b")

        testlogger = TestLogger()
        p.setup(logger=testlogger)

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 1)

        self.assertEqual(warnings[0] ,"Group '' has the following cycles: [['G1', 'C4']]")

        # test recursive cycle check
        sccs = get_cycles(root, recurse=True)
        self.assertEqual([['G1.C1', 'G1.C2', 'C4']], sccs)


if __name__ == "__main__":
    unittest.main()
