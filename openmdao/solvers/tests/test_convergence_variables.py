import unittest

import numpy as np

from openmdao.api import Problem, ExplicitComponent, NonlinearBlockGS, Group, ScipyKrylov, IndepVarComp, \
    ExecComp, AnalysisError
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives, \
    SellarDis1, SellarDis2
from random import random

class ContrivedSellarDis1(SellarDis1):

    def setup(self):
        super(ContrivedSellarDis1, self).setup()
        self.add_output('highly_nonlinear', val=1.0)

    def compute(self, inputs, outputs):
        super(ContrivedSellarDis1, self).compute(inputs, outputs)
        outputs['highly_nonlinear'] = 10*np.sin(10*inputs['y2'])

class TestConvergenceVariables(unittest.TestCase):

    def test_convergence_variables(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', ContrivedSellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        nlbgs = model.nonlinear_solver = NonlinearBlockGS()

        nlbgs.options['maxiter'] = 20
        nlbgs.options['atol'] = 1e-6
        nlbgs.options['rtol'] = 1e-6
        nlbgs.options['iprint'] = 2

        prob.setup()
        prob.run_model()
        nb1 = nlbgs._iter_count

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        nlbgs.options['convrg_vars'] = ['d1.y1', 'd2.y2']

        prob.setup()
        prob.run_model()
        nb2 = nlbgs._iter_count
        self.assertGreater(nb1, nb2)

        nlbgs.options['convrg_rtols'] = [1e-3, 1e-3]

        prob.setup()
        prob.run_model()
        nb3 = nlbgs._iter_count
        self.assertGreater(nb2, nb3)

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)


if __name__ == "__main__":
    unittest.main()