"""Test the LinearBlockJac class."""

from __future__ import division, print_function

import unittest

import numpy as np

from openmdao.api import Group, IndepVarComp, Problem, LinearBlockJac, AssembledJacobian, \
    ExecComp, NonlinearBlockGS
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.solvers.linear.tests.linear_test_base import LinearSolverTests


class TestLinearBlockJacSolver(LinearSolverTests.LinearSolverTestCase):

    linear_solver_class = LinearBlockJac

    def test_globaljac_err(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.linear_solver = LinearBlockJac()
        prob.set_solver_print(level=0)

        prob.model.jacobian = AssembledJacobian()
        prob.setup(check=False, mode='fwd')

        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        with self.assertRaises(RuntimeError) as context:
            prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

            self.assertEqual(str(context.exception),
                             "A block linear solver 'LN: LNBJ' is being used with"
                             " an AssembledJacobian in system ''")


class TestBJacSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockJac, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

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
        model.linear_solver = LinearBlockJac()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockJac, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

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

        model.linear_solver = LinearBlockJac()
        model.linear_solver.options['maxiter'] = 5

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.60230118004, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78022500547, .00001)

    def test_feature_atol(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockJac, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

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

        model.linear_solver = LinearBlockJac()
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

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockJac, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

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

        model.linear_solver = LinearBlockJac()
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
