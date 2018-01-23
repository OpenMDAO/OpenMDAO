"""Test the Nonlinear Block Jacobi solver. """

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, LinearBlockGS
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives


class TestNLBlockJacobi(unittest.TestCase):

    def test_feature_basic(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, NonlinearBlockJac, LinearBlockGS
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

        model.linear_solver = LinearBlockGS()
        model.nonlinear_solver = NonlinearBlockJac()

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_maxiter(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, NonlinearBlockJac, LinearBlockGS
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

        model.linear_solver = LinearBlockGS()

        nlgbs = model.nonlinear_solver = NonlinearBlockJac()
        nlgbs.options['maxiter'] = 4

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5723813937, .00001)
        assert_rel_error(self, prob['y2'], 12.0542542372, .00001)

    def test_feature_rtol(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, NonlinearBlockJac, LinearBlockGS
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

        model.linear_solver = LinearBlockGS()

        nlgbs = model.nonlinear_solver = NonlinearBlockJac()
        nlgbs.options['rtol'] = 1e-3

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5891491526, .00001)
        assert_rel_error(self, prob['y2'], 12.0569142166, .00001)

    def test_feature_atol(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, NonlinearBlockJac, LinearBlockGS
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

        model.linear_solver = LinearBlockGS()

        nlgbs = model.nonlinear_solver = NonlinearBlockJac()
        nlgbs.options['atol'] = 1e-2

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5886171567, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()
