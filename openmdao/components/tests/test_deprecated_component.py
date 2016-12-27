from __future__ import print_function

import unittest

from openmdao.api import Problem, IndepVarComp, Component, Group, NewtonSolver
from openmdao.api import GlobalJacobian, DenseMatrix, ScipyIterativeSolver

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

        J['z2', 'x'] = 2.
        J['z2', 'z1'] = 2

        return J


class DepCompTestCase(unittest.TestCase):

    def test_run(self):
        group = Group()
        group.add_subsystem('sys1', IndepVarComp('x', val=4.))
        group.add_subsystem('sys2', IndepVarComp('y', val=3.))
        group.add_subsystem('sys3', TestComp())

        p = Problem()
        p.root = group
        p.setup(check=False)
        p.root.suppress_solver_output = True

        p.run()

        assert_rel_error(self, p['sys3.z1'], 8., 1e-10)
        assert_rel_error(self, p['sys3.z2'], 24, 1e-10)

    def test_run_with_linearize(self):
        group = Group()
        group.add_subsystem('sys1', IndepVarComp('x', val=4.))
        group.add_subsystem('sys2', IndepVarComp('y', val=3.))
        group.add_subsystem('sys3', TestComp())

        p = Problem()
        p.root = group
        p.root.nl_solver = NewtonSolver(
            subsolvers={'linear': ScipyIterativeSolver()})
        p.setup(check=False)
        p.root.suppress_solver_output = True

        #p.root.jacobian = GlobalJacobian(matrix_class=DenseMatrix)
        #print(p.root.jacobian._int_mtx._matrix)

        p.run()

        assert_rel_error(self, p['sys3.z1'], 8., 1e-10)
        assert_rel_error(self, p['sys3.z2'], 24, 1e-10)


if __name__ == "__main__":
    unittest.main()
