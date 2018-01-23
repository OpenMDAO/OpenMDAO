"""Test the Nonlinear Block Gauss Seidel solver. """

import unittest

import numpy as np

from openmdao.api import Problem, NonlinearBlockGS, Group, ScipyKrylov, IndepVarComp, \
     ExecComp, AnalysisError
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives, \
     SellarDis1withDerivatives, SellarDis2withDerivatives


class TestNLBGaussSeidel(unittest.TestCase):

    def test_feature_set_options(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp, NonlinearBlockGS
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

        nlgbs = model.nonlinear_solver = NonlinearBlockGS()

        nlgbs.options['maxiter'] = 20
        nlgbs.options['atol'] = 1e-6
        nlgbs.options['rtol'] = 1e-6

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_basic(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp, NonlinearBlockGS
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

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_maxiter(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp, NonlinearBlockGS
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

        nlgbs = model.nonlinear_solver = NonlinearBlockGS()
        nlgbs.options['maxiter'] = 2

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5454858939, .00001)
        assert_rel_error(self, prob['y2'], 12.0542542372, .00001)

    def test_feature_rtol(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives, SellarDerivatives

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

        nlgbs = model.nonlinear_solver = NonlinearBlockGS()
        nlgbs.options['rtol'] = 1e-3

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5891491526, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_atol(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp, NonlinearBlockGS
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

        nlgbs = model.nonlinear_solver = NonlinearBlockGS()
        nlgbs.options['atol'] = 1e-4

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5882856302, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_sellar(self):
        # Basic sellar test.

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

        nlgbs = model.nonlinear_solver = NonlinearBlockGS()

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(model.nonlinear_solver._iter_count, 8)

        # Make sure we only call apply_linear on 'heads'
        #nd1 = model.cycle.d1.execution_count
        #nd2 = model.cycle.d2.execution_count
        #if model.cycle.d1._run_apply == True:
            #self.assertEqual(nd1, 2*nd2)
        #else:
            #self.assertEqual(2*nd1, nd2)

    def test_sellar_analysis_error(self):
        # Tests Sellar behavior when AnalysisError is raised.

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

        nlgbs = model.nonlinear_solver = NonlinearBlockGS()
        nlgbs.options['maxiter'] = 2
        nlgbs.options['err_on_maxiter'] = True

        prob.setup(check=False)
        prob.set_solver_print(level=0)

        try:
            prob.run_model()
        except AnalysisError as err:
            self.assertEqual(str(err), "Solver 'NL: NLBGS' on system '' failed to converge.")
        else:
            self.fail("expected AnalysisError")

    def test_sellar_group_nested(self):
        # This tests true nested gs. Subsolvers solve each Sellar system. Top
        # solver couples them together through variable x.

        # This version has the indepvarcomps removed so we can connect them together.
        class SellarModified(Group):
            """ Group containing the Sellar MDA. This version uses the disciplines
            with derivatives."""

            def __init__(self):
                super(SellarModified, self).__init__()

                self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

                self.nonlinear_solver = NonlinearBlockGS()
                self.linear_solver = ScipyKrylov()

        prob = Problem()
        root = prob.model
        root.nonlinear_solver = NonlinearBlockGS()
        root.nonlinear_solver.options['maxiter'] = 20
        root.add_subsystem('g1', SellarModified())
        root.add_subsystem('g2', SellarModified())

        root.connect('g1.y2', 'g2.x')
        root.connect('g2.y2', 'g1.x')

        prob.setup(check=False)
        prob.set_solver_print(level=0)

        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)

    def test_NLBGS_Aitken(self):

        prob = Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()
        model.nonlinear_solver.options['use_aitken'] = True
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)
        self.assertTrue(model.nonlinear_solver._iter_count == 5)

if __name__ == "__main__":
    unittest.main()
