"""Test the LinearBlockGS linear solver class."""

from __future__ import division, print_function
import unittest

from openmdao.solvers.tests.linear_test_base import LinearSolverTests
from openmdao.devtools.testutil import assert_rel_error
from openmdao.api import LinearBlockGS, Problem, Group, ImplicitComponent, IndepVarComp, \
    DirectSolver, NewtonSolver, ScipyIterativeSolver, AssembledJacobian
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarImplicitDis1, SellarImplicitDis2
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense


class TestBGSSolver(LinearSolverTests.LinearSolverTestCase):
    ln_solver_class = LinearBlockGS

    def test_globaljac_err(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = self.ln_solver_class()
        prob.set_solver_print(level=0)

        prob.model.jacobian = AssembledJacobian()
        prob.setup(check=False, mode='fwd')

        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        with self.assertRaises(RuntimeError) as context:
            prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')

        self.assertEqual(str(context.exception),
                         "A block linear solver 'LN: LNBGS' is being used with"
                         " an AssembledJacobian in system ''")

    def test_simple_implicit(self):
        # This verifies that we can perform lgs around an implicit comp and get the right answer
        # as long as we slot a non-lgs linear solver on that component.

        class SimpleImp(ImplicitComponent):

            def initialize_variables(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = 3.0*inputs['a'] + 2.0*outputs['x']

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = 2.0
                jacobian['x', 'a'] = 3.0

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p', IndepVarComp('a', 5.0))
        comp = model.add_subsystem('comp', SimpleImp())
        model.connect('p.a', 'comp.a')

        model.ln_solver = self.ln_solver_class()
        comp.ln_solver = DirectSolver()

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        deriv = prob.compute_total_derivs(of=['comp.x'], wrt=['p.a'])
        self.assertEqual(deriv['comp.x', 'p.a'], -1.5)

    def test_implicit_cycle(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 1.0))
        model.add_subsystem('d1', SellarImplicitDis1())
        model.add_subsystem('d2', SellarImplicitDis2())
        model.connect('d1.y1', 'd2.y1')
        model.connect('d2.y2', 'd1.y2')

        model.nl_solver = NewtonSolver()
        model.nl_solver.options['maxiter'] = 5
        model.ln_solver = self.ln_solver_class()

        prob.setup(check=False)
        prob.set_solver_print(level=0)

        prob.run_model()
        res = model._residuals.get_norm()

        # Newton is kinda slow on this for some reason, this is how far it gets with directsolver too.
        self.assertLess(res, 2.0e-2)

    def test_implicit_cycle_precon(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 1.0))
        model.add_subsystem('d1', SellarImplicitDis1())
        model.add_subsystem('d2', SellarImplicitDis2())
        model.connect('d1.y1', 'd2.y1')
        model.connect('d2.y2', 'd1.y2')

        model.nl_solver = NewtonSolver()
        model.nl_solver.options['maxiter'] = 5
        model.ln_solver = ScipyIterativeSolver()
        model.ln_solver.precon = self.ln_solver_class()

        prob.setup(check=False)
        prob.set_solver_print(level=0)

        prob.run_model()
        res = model._residuals.get_norm()

        # Newton is kinda slow on this for some reason, this is how far it gets with directsolver too.
        self.assertLess(res, 2.0e-2)

class TestBGSSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockGS()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockGS()
        model.ln_solver.options['maxiter'] = 2

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.60230118004, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78022500547, .00001)

    def test_feature_atol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockGS()
        model.ln_solver.options['atol'] = 1.0e-3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78456955704, .00001)

    def test_feature_rtol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockGS()
        model.ln_solver.options['rtol'] = 1.0e-3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78456955704, .00001)


if __name__ == "__main__":
    unittest.main()
