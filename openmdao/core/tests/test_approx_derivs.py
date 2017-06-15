""" Testing for group finite differencing."""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ScipyIterativeSolver, ExecComp, NewtonSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray, TestImplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.simple_comps import DoubleArrayComp


class TestGroupFiniteDifference(unittest.TestCase):

    def test_paraboloid(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyIterativeSolver()
        model.approx_total_derivs()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        # 1 output x 2 inputs
        self.assertEqual(len(model._approx_schemes['fd']._exec_list), 2)

    def test_paraboloid_subbed(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', Group(), promotes=['x', 'y', 'f_xy'])
        sub.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyIterativeSolver()
        sub.approx_total_derivs()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        Jfd = sub.jacobian._subjacs
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.x'], [[6.0]], 1e-6)
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.y'], [[-8.0]], 1e-6)

        # 1 output x 2 inputs
        sub = model.get_subsystem('sub')
        self.assertEqual(len(sub._approx_schemes['fd']._exec_list), 2)

    def test_paraboloid_subbed_with_connections(self):
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

        model.linear_solver = ScipyIterativeSolver()
        sub.approx_total_derivs()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['sub.comp.f_xy']
        wrt = ['p1.x', 'p2.y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['sub.comp.f_xy', 'p1.x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['sub.comp.f_xy', 'p2.y'], [[8.0]], 1e-6)

        Jfd = sub.jacobian._subjacs
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.bx.xin'], [[6.0]], 1e-6)
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.by.yin'], [[-8.0]], 1e-6)

        # 3 outputs x 2 inputs
        sub = model.get_subsystem('sub')
        self.assertEqual(len(sub._approx_schemes['fd']._exec_list), 6)

    def test_arrray_comp(self):

        class DoubleArrayFD(DoubleArrayComp):

            def compute_partials(self, inputs, outputs, partials):
                """
                Override deriv calculation.
                """
                pass

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x1', val=np.ones(2)))
        model.add_subsystem('p2', IndepVarComp('x2', val=np.ones(2)))
        comp = model.add_subsystem('comp', DoubleArrayFD())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        model.linear_solver = ScipyIterativeSolver()
        model.approx_total_derivs()

        prob.setup(check=False)
        prob.run_model()
        model.run_linearize()

        Jfd = model.jacobian._subjacs
        assert_rel_error(self, Jfd['comp.y1', 'p1.x1'], -comp.JJ[0:2, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y1', 'p2.x2'], -comp.JJ[0:2, 2:4], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p1.x1'], -comp.JJ[2:4, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p2.x2'], -comp.JJ[2:4, 2:4], 1e-6)

    def test_implicit_component_fd(self):
        # Somehow this wasn't tested in the original fd tests (which are mostly feature tests.)

        class TestImplCompArrayDense(TestImplCompArray):

            def setup_partials(self):
                self.approx_partials('*', '*')

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p_rhs', IndepVarComp('rhs', val=np.ones(2)))
        sub = model.add_subsystem('sub', Group())
        comp = sub.add_subsystem('comp', TestImplCompArrayDense())
        model.connect('p_rhs.rhs', 'sub.comp.rhs')

        model.linear_solver = ScipyIterativeSolver()

        prob.setup(check=False)
        prob.run_model()
        model.run_linearize()

        Jfd = comp.jacobian._subjacs
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.rhs'], -np.eye(2), 1e-6)
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.x'], comp.mtx, 1e-6)

    def test_around_newton(self):
        # For a group that is set to FD that has a Newton solver, make sure it doesn't
        # try to FD itself while solving.

        class TestImplCompArrayDenseNoSolve(TestImplCompArrayDense):
            def solve_nonlinear(self, inputs, outputs):
                """ Disable local solve."""
                pass


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p_rhs', IndepVarComp('rhs', val=np.array([2, 4])))
        comp = model.add_subsystem('comp', TestImplCompArrayDenseNoSolve())
        model.connect('p_rhs.rhs', 'comp.rhs')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyIterativeSolver()
        model.approx_total_derivs()

        prob.setup(check=False)
        prob.run_model()
        model.approx_total_derivs()
        assert_rel_error(self, prob['comp.x'], [1.97959184, 4.02040816], 1e-5)

        model.run_linearize()

        of = ['comp.x']
        wrt = ['p_rhs.rhs']
        Jfd = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, Jfd['comp.x', 'p_rhs.rhs'], [[1.01020408, -0.01020408], [-0.01020408,  1.01020408]], 1e-5)

    def test_step_size(self):
        # Test makes sure option metadata propagates to the fd function
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyIterativeSolver()

        # Worse step so that our answer will be off a wee bit.
        model.approx_total_derivs(step=1e-2)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-5.99]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.01]], 1e-6)

if __name__ == "__main__":
    unittest.main()
