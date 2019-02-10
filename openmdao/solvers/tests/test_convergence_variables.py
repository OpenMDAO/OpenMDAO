import unittest
import numpy as np

from openmdao.api import Problem, IndepVarComp, NonlinearBlockGS
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2
from openmdao.utils.assert_utils import assert_rel_error

class ContrivedSellarDis1(SellarDis1):

    def setup(self):
        super(ContrivedSellarDis1, self).setup()
        self.add_output('highly_nonlinear', val=1.0)

    def compute(self, inputs, outputs):
        super(ContrivedSellarDis1, self).compute(inputs, outputs)
        outputs['highly_nonlinear'] = 10*np.sin(10*inputs['y2'])

class TestConvergenceVariables(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()
        model = self.prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', ContrivedSellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        self.nlbgs = nlbgs = model.nonlinear_solver = NonlinearBlockGS()

        nlbgs.options['maxiter'] = 20
        nlbgs.options['atol'] = 1e-6
        nlbgs.options['rtol'] = 1e-6
        nlbgs.options['iprint'] = 2

    def test_convergence_variables(self):
        prob = self.prob
        nlbgs = self.nlbgs

        prob.setup()
        prob.run_model()
        nb1 = nlbgs._iter_count

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        nlbgs.options['convrg_vars'] = ['d1.y1', 'd2.y2']

        prob.setup()
        prob.run_model()
        nb2 = nlbgs._iter_count
        self.assertLess(nb2, nb1)

        nlbgs.options['convrg_rtols'] = [1e-3, 1e-3]

        prob.setup()
        prob.run_model()
        nb3 = nlbgs._iter_count
        self.assertLess(nb3, nb2)

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_bad_size(self):
        self.nlbgs.options['convrg_vars'] = ['d1.y1', 'd2.y2']
        self.nlbgs.options['convrg_rtols'] = [1e-3]

        self.prob.setup()
        with self.assertRaises(RuntimeError) as context:
            self.prob.run_model()
        self.assertEqual(str(context.exception),
                         "Convergence rtols bad size : should be 2, found 1.")

if __name__ == "__main__":
    unittest.main()
