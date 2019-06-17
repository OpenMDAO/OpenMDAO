"""Test the LinearBlockGS linear solver class."""

from __future__ import division, print_function
import unittest

import numpy as np

from openmdao.solvers.linear.tests.linear_test_base import LinearSolverTests
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.api import LinearBlockGS, Problem, Group, ImplicitComponent, IndepVarComp, \
    DirectSolver, NewtonSolver, ScipyKrylov, ExecComp, NonlinearBlockGS, BoundsEnforceLS
from openmdao.test_suite.components.sellar import SellarImplicitDis1, SellarImplicitDis2, \
    SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.sellar import SellarDerivatives


class SimpleImp(ImplicitComponent):
    def setup(self):
        self.add_input('a', val=1.)
        self.add_output('x', val=0.)

        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = 3.0*inputs['a'] + 2.0*outputs['x']

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = 2.0
        jacobian['x', 'a'] = 3.0


class TestBGSSolver(LinearSolverTests.LinearSolverTestCase):
    linear_solver_class = LinearBlockGS

    def test_globaljac_err(self):
        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.linear_solver = self.linear_solver_class(assemble_jac=True)
        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.run_model()

        self.assertEqual(str(context.exception),
                         "Linear solver 'LN: LNBGS' in system '' doesn't support assembled jacobians.")

    def test_simple_implicit(self):
        # This verifies that we can perform lgs around an implicit comp and get the right answer
        # as long as we slot a non-lgs linear solver on that component.

        prob = Problem()
        model = prob.model
        model.add_subsystem('p', IndepVarComp('a', 5.0))
        comp = model.add_subsystem('comp', SimpleImp())
        model.connect('p.a', 'comp.a')

        model.linear_solver = self.linear_solver_class()
        comp.linear_solver = DirectSolver()

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        deriv = prob.compute_totals(of=['comp.x'], wrt=['p.a'])
        self.assertEqual(deriv['comp.x', 'p.a'], -1.5)

    def test_implicit_cycle(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 1.0))
        model.add_subsystem('d1', SellarImplicitDis1())
        model.add_subsystem('d2', SellarImplicitDis2())
        model.connect('d1.y1', 'd2.y1')
        model.connect('d2.y2', 'd1.y2')

        model.nonlinear_solver = NewtonSolver()
        model.nonlinear_solver.options['maxiter'] = 5
        model.linear_solver = self.linear_solver_class()

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        res = model._residuals.get_norm()

        # Newton is kinda slow on this for some reason, this is how far it gets with directsolver too.
        self.assertLess(res, 2.0e-2)

    def test_implicit_cycle_precon(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 1.0))
        model.add_subsystem('d1', SellarImplicitDis1())
        model.add_subsystem('d2', SellarImplicitDis2())
        model.connect('d1.y1', 'd2.y1')
        model.connect('d2.y2', 'd1.y2')

        model.nonlinear_solver = NewtonSolver()
        model.nonlinear_solver.options['maxiter'] = 5
        model.nonlinear_solver.linesearch = BoundsEnforceLS()
        model.linear_solver = ScipyKrylov()
        model.linear_solver.precon = self.linear_solver_class()

        prob.setup()

        prob['d1.y1'] = 4.0
        prob.set_solver_print()
        prob.run_model()
        res = model._residuals.get_norm()

        # Newton is kinda slow on this for some reason, this is how far it gets with directsolver too.
        self.assertLess(res, 2.0e-2)

    def test_full_desvar_with_index_obj_relevance_bug(self):
        prob = Problem()
        sub = prob.model.add_subsystem('sub', SellarDerivatives())
        prob.model.nonlinear_solver = NonlinearBlockGS()
        prob.model.linear_solver = LinearBlockGS()
        sub.nonlinear_solver = NonlinearBlockGS()
        sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('sub.z', lower=-100, upper=100)
        prob.model.add_objective('sub.z', index=1)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        derivs = prob.compute_totals(of=['sub.z'], wrt=['sub.z'])

        assert_rel_error(self, derivs[('sub.z', 'sub.z')], [[0., 1.]])


class TestBGSSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockGS, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = LinearBlockGS()
        model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockGS, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = LinearBlockGS()
        model.linear_solver.options['maxiter'] = 2

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.60230118004, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78022500547, .00001)

    def test_feature_atol(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockGS, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = LinearBlockGS()
        model.linear_solver.options['atol'] = 1.0e-3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78456955704, .00001)

    def test_feature_rtol(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockGS, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = LinearBlockGS()
        model.linear_solver.options['rtol'] = 1.0e-3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78456955704, .00001)


if __name__ == "__main__":
    unittest.main()
