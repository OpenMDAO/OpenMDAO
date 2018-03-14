""" Test the ExternalCode. """
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, ExecComp, ScipyOptimizeDriver, \
     ExplicitComponent
from openmdao.components.ks import KSComp
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.utils.assert_utils import assert_rel_error


class TestExternalCode(unittest.TestCase):

    def test_basic_ks(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))
        model.add_subsystem('comp', DoubleArrayComp())
        model.add_subsystem('ks', KSComp(width=2))
        model.connect('px.x', 'comp.x1')
        model.connect('comp.y2', 'ks.g')

        model.add_design_var('px.x')
        model.add_objective('comp.y1')
        model.add_constraint('ks.KS', upper=0.0)

        prob.setup(check=False)
        prob.run_driver()

        assert_rel_error(self, max(prob['comp.y2']), prob['ks.KS'])

    def test_ks_opt_sellar(self):

        class Mux(ExplicitComponent):

            def setup(self):
                self.add_input('g1', 0.0)
                self.add_input('g2', 0.0)
                self.add_output('g', np.zeros((2, )))

                self.declare_partials(of='g', wrt='g1', val=np.array([[1.0], [0.0]]))
                self.declare_partials(of='g', wrt='g2', val=np.array([[0.0], [1.0]]))

            def compute(self, inputs, outputs):
                outputs['g'][0] = inputs['g1']
                outputs['g'][1] = inputs['g2']

        prob = Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = "SLSQP"

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')

        model.add_subsystem('cons', Mux())
        model.add_subsystem('ks', KSComp(width=2))
        model.connect('con1', 'cons.g1')
        model.connect('con2', 'cons.g2')
        model.connect('cons.g', 'ks.g')

        #model.add_constraint('ks.KS', upper=0.0)
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_rel_error(self, prob['z'][0], 1.98337708, 1e-3)

if __name__ == "__main__":
    unittest.main()
