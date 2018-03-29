""" Test the KSFunction component. """
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, ExecComp, ScipyOptimizeDriver, \
     ExplicitComponent
from openmdao.components.ks import KSComponent
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup
from openmdao.utils.assert_utils import assert_rel_error


class TestKSFunction(unittest.TestCase):

    def test_basic_ks(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))
        model.add_subsystem('comp', DoubleArrayComp())
        model.add_subsystem('ks', KSComponent(width=2))
        model.connect('px.x', 'comp.x1')
        model.connect('comp.y2', 'ks.g')

        model.add_design_var('px.x')
        model.add_objective('comp.y1')
        model.add_constraint('ks.KS', upper=0.0)

        prob.setup(check=False)
        prob.run_driver()

        assert_rel_error(self, max(prob['comp.y2']), prob['ks.KS'])

    def test_beam_stress(self):
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        max_bending = 100.0

        num_cp = 5
        num_elements = 25
        num_load_cases = 2

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume, max_bending = max_bending,
                                                 num_elements=num_elements, num_cp=num_cp,
                                                 num_load_cases=num_load_cases))

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        prob.setup(mode='rev')

        prob.run_driver()

        stress0 = prob['parallel.sub_0.stress_comp.stress_0']
        stress1 = prob['parallel.sub_0.stress_comp.stress_1']

        # Test that the the maximum constraint prior to aggregation is close to "active".
        assert_rel_error(self, max(stress0), 100.0, tolerance=5e-2)
        assert_rel_error(self, max(stress1), 100.0, tolerance=5e-2)

        # Test that no original constraint is violated.
        self.assertTrue(np.all(stress0 < 100.0))
        self.assertTrue(np.all(stress1 < 100.0))

    def test_upper(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', KSComponent(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['upper'] = 16.0
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'], -1.0)

    def test_lower_flag(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', KSComponent(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['lower_flag'] = True
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'], -12.0)


class TestKSFunctionFeatures(unittest.TestCase):

    def test_basic(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp
        from openmdao.components.ks import KSComponent

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', KSComponent(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'], 15.0)

    def test_upper(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp
        from openmdao.components.ks import KSComponent

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', KSComponent(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['upper'] = 16.0
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'], -1.0)

    def test_lower_flag(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp
        from openmdao.components.ks import KSComponent

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', KSComponent(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['lower_flag'] = True
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'], -12.0)

if __name__ == "__main__":
    unittest.main()
