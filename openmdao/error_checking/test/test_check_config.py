import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.devtools.testutil import TestLogger

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
        p.setup(check=True, logger=testlogger)

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


if __name__ == "__main__":
    unittest.main()
