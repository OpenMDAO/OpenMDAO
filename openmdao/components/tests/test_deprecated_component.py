from __future__ import print_function

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

        if mode == 'fwd':

            if 'x' in dp:
                if 'z2' in dr:
                    dr['z2'] = 2.0*dp['x']

            if 'y' in dp:
                if 'z1' in dr:
                    dr['z1'] = dp['y']

            if 'z1' in du:
                if 'z2' in dr:
                    dr['z2'] = 2.0*dp['z1']

        elif mode == 'rev':

            if 'x' in dp:
                if 'z2' in dr:
                    dp['x'] = 2.0*dr['z2']

            if 'y' in dp:
                if 'z1' in dr:
                    dp['y'] = dr['z1']

            if 'z1' in du:
                if 'z2' in dr:
                    dp['z1'] = 2.0*dr['z2']


class DepCompTestCase(unittest.TestCase):

    def test_run_model(self):
        group = Group()
        group.add('sys1', IndepVarComp('x', val=4.))
        group.add('sys2', IndepVarComp('y', val=3.))
        group.add('sys3', TestComp())

        if openmdao_version == 1:
            params, unknowns, dunknowns = {}, {}, {}
            dparams = {'x': 4., 'y': 3}
            dresids = {'z1': 0., 'z2': 0.}
            group.sys3.apply_linear(params, unknowns, dparams, dunknowns, dresids, 'fwd')
            print('J:', group.sys3._jacobian_cache)

        p = Problem()
        p.root = group
        if openmdao_version == 1:
            p.root.ln_solver = ScipyGMRES()
        p.setup(check=False)
        p.root.suppress_solver_output = True

        p.run()

        assert_rel_error(self, p['sys3.z1'], 8., 1e-10)
        assert_rel_error(self, p['sys3.z2'], 24, 1e-10)

        if openmdao_version == 2:
            J = p.compute_total_derivs(of=['sys3.z1', 'sys3.z2'],
                                       wrt=['sys1.x', 'sys2.y'])
        else:
            # OpenMDAO 1.x
            J = p.check_total_derivatives()
        print(J)

    def test_run_with_linearize(self):
        group = Group()
        group.add('sys1', IndepVarComp('x', val=4.))
        group.add('sys2', IndepVarComp('y', val=3.))
        group.add('sys3', TestComp())

        p = Problem()
        p.root = group
        p.root.nl_solver.ln_solver = ScipyIterativeSolver()
        if openmdao_version == 1:
            p.root.ln_solver = ScipyGMRES()
        p.setup(check=False)
        p.root.suppress_solver_output = True

        p.run()

        assert_rel_error(self, p['sys3.z1'], 8., 1e-10)
        assert_rel_error(self, p['sys3.z2'], 24, 1e-10)

        if openmdao_version == 2:
            J = p.compute_total_derivs(of=['sys3.z1', 'sys3.z2'],
                                       wrt=['sys1.x', 'sys2.y'])
        else:
            J = p.check_total_derivatives()
        print(J)


if __name__ == "__main__":
    unittest.main()
