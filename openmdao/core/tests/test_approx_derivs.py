""" Testing for group finite differencing."""

import unittest

from openmdao.api import Problem, Group, IndepVarComp, ScipyIterativeSolver, ExecComp
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid


class TestGroupFiniteDifference(unittest.TestCase):

    def test_paraboloid(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.ln_solver = ScipyIterativeSolver()
        model.approx_all_partials()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

    def test_paraboloid_subbed(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', Group(), promotes=['x', 'y', 'f_xy'])
        sub.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.ln_solver = ScipyIterativeSolver()
        sub.approx_all_partials()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

    def test_paraboloid_subbed_with_conns(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0))
        model.add_subsystem('p2', IndepVarComp('y', 0.0))
        sub = model.add_subsystem('sub', Group())
        sub.add_subsystem('bx', ExecComp('xout = xin'))
        sub.add_subsystem('by', ExecComp('yout = yin'))
        sub.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'sub.bx.xin')
        model.connect('sub.bx.xout', 'sub.comp.x')
        model.connect('p2.y', 'sub.by.yin')
        model.connect('sub.by.yout', 'sub.comp.y')

        model.ln_solver = ScipyIterativeSolver()
        sub.approx_all_partials()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['sub.comp.f_xy']
        wrt = ['p1.x', 'p2.y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['sub.comp.f_xy', 'p1.x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['sub.comp.f_xy', 'p2.y'], [[8.0]], 1e-6)

if __name__ == "__main__":
    unittest.main()
