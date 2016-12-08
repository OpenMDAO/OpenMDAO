from __future__ import print_function

import unittest

from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.components.deprecated_component import Component

from openmdao.devtools.testutil import assert_rel_error


class TestComp(Component):

    def __init__(self):
        super(TestComp, self).__init__()

        self.add_param('x', val=4.)
        self.add_param('y', val=3.)
        self.add_state('z1', val=0.)
        self.add_output('z2', val=0.)

    def apply_nonlinear(self, p, u, r):

        r['z1'] = 5. - u['z1']+p['y']

        r['z2'] = u['z2'] - (2*p['x'] + 2*u['z1'])

    def solve_nonlinear(self, p, u, r):

        u['z1'] = 5. + p['y']
        u['z2'] = 2*p['x'] + 2*u['z1']

    def linearize(self, p, u, r):

        J = {}
        J['z1', 'y'] = 1.
        J['z1', 'z1'] = -1.

        J['z2', 'x'] = 2
        J['z2', 'z1'] = 2


class DepCompTestCase(unittest.TestCase):

    def test_run_with_linearize(self):

        p = Problem()
        p.root = TestComp()
        p.setup()

        p.run()

        assert_rel_error(self, p['z1'], 8., 1e-10)
        assert_rel_error(self, p['z2'], 24, 1e-10)


if __name__ == "__main__":
    unittest.main()
