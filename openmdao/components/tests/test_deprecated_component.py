from __future__ import print_function
from pprint import pprint

import unittest

from openmdao.api import Problem, IndepVarComp, Component, Group

try:
    # OpenMDAO 2.x
    from openmdao.api import ScipyIterativeSolver
    from openmdao.devtools.testutil import assert_rel_error
    openmdao_version = 2
except ImportError:
    # OpenMDAO 1.x
    from openmdao.api import ScipyOptimizer as ScipyIterativeSolver
    from openmdao.test.util import assert_rel_error
    from openmdao.solvers.scipy_gmres import ScipyGMRES
    openmdao_version = 1


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


class TestCompApply(TestComp):

    def apply_linear(self, p, u, dp, du, dr, mode):
        print('TestCompApply.apply_linear()\n----------------------------')
        if openmdao_version == 1:
            vals = list(p[key] for key in p.keys())
            print(vals)
            print('p:', p.keys(), list(p[key] for key in p.keys()))
            print('u:', u.keys(), list(u[key] for key in u.keys()))
            print('dp:', dp.keys(), list(dp[key] for key in dp.keys()))
            print('du:', du.keys(), list(du[key] for key in du.keys()))
            print('dr:', dr.keys(), list(dr[key] for key in dr.keys()))

        if openmdao_version == 2:
            print('p:', p._names, p.get_data())
            print('u:', u._names, u.get_data())
            print('dp:', dp._names, dp.get_data())
            print('du:', du._names, du.get_data())
            print('dr:', dr._names, dr.get_data())

        if mode == 'fwd':
            if 'x' in dp:
                if 'z2' in dr:
                    dr['z2'] = 2.0*dp['x']
            if 'y' in dp:
                if 'z1' in dr:
                    dr['z1'] = dp['y']
            if 'z1' in du:
                if 'z2' in dr:
                    dr['z2'] = 2.0*du['z1']
        elif mode == 'rev':
            if 'x' in dp:
                if 'z2' in dr:
                    dp['x'] = 2.0*dr['z2']
            if 'y' in dp:
                if 'z1' in dr:
                    dp['y'] = dr['z1']
            if 'z1' in du:
                if 'z2' in dr:
                    du['z1'] = 2.0*dr['z2']


class DepCompTestCase(unittest.TestCase):

    def test_run_model(self):
        p = Problem(Group())

        p.root.add('sys1', IndepVarComp('x', val=4.), promotes=['x'])
        p.root.add('sys2', IndepVarComp('y', val=3.), promotes=['y'])
        p.root.add('sys3', TestCompApply(), promotes=['x', 'y'])

        if openmdao_version == 1:
            p.root.ln_solver = ScipyGMRES()

        p.setup(check=False)
        # p.root.suppress_solver_output = True
        p.run()

        assert_rel_error(self, p['sys3.z1'], 8., 1e-10)
        assert_rel_error(self, p['sys3.z2'], 24, 1e-10)

        if openmdao_version == 1:
            J = p.calc_gradient(['x', 'y'], ['sys3.z1', 'sys3.z2'])

        if openmdao_version == 2:
            J = p.compute_total_derivs(of=['sys3.z1', 'sys3.z2'],
                                       wrt=['x', 'y'])

        print('Jacobian:')
        pprint(J)

    def test_run_with_linearize(self):
        p = Problem(Group())

        p.root.add('sys1', IndepVarComp('x', val=4.), promotes=['x'])
        p.root.add('sys2', IndepVarComp('y', val=3.), promotes=['y'])
        p.root.add('sys3', TestCompApply(), promotes=['x', 'y'])

        p.root.nl_solver.ln_solver = ScipyIterativeSolver()
        if openmdao_version == 1:
            p.root.ln_solver = ScipyGMRES()

        p.setup(check=False)
        # p.root.suppress_solver_output = True
        p.run()

        assert_rel_error(self, p['sys3.z1'], 8., 1e-10)
        assert_rel_error(self, p['sys3.z2'], 24, 1e-10)

        if openmdao_version == 1:
            J = p.calc_gradient(['x', 'y'], ['sys3.z1', 'sys3.z2'])

        if openmdao_version == 2:
            J = p.compute_total_derivs(of=['sys3.z1', 'sys3.z2'],
                                       wrt=['x', 'y'])

        print('Jacobian:')
        pprint(J)


if __name__ == "__main__":
    unittest.main()
