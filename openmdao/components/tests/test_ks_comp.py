""" Test the KSFunction component. """
from __future__ import print_function

import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup
from openmdao.utils.assert_utils import assert_rel_error, assert_warning


class TestKSFunction(unittest.TestCase):

    def test_basic_ks(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        model.add_subsystem('comp', DoubleArrayComp())
        model.add_subsystem('ks', om.KSComp(width=2))
        model.connect('px.x', 'comp.x1')
        model.connect('comp.y2', 'ks.g')

        model.add_design_var('px.x')
        model.add_objective('comp.y1')
        model.add_constraint('ks.KS', upper=0.0)

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, max(prob['comp.y2']), prob['ks.KS'][0])

    def test_vectorized(self):
        prob = om.Problem()
        model = prob.model

        x = np.zeros((3, 5))
        x[0, :] = np.array([3.0, 5.0, 11.0, 13.0, 17.0])
        x[1, :] = np.array([13.0, 11.0, 5.0, 17.0, 3.0])*2
        x[2, :] = np.array([11.0, 3.0, 17.0, 5.0, 13.0])*3

        model.add_subsystem('px', om.IndepVarComp(name="x", val=x))
        model.add_subsystem('ks', om.KSComp(width=5, vec_size=3))
        model.connect('px.x', 'ks.g')

        model.add_design_var('px.x')
        model.add_constraint('ks.KS', upper=0.0)

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, prob['ks.KS'][0], 17.0)
        assert_rel_error(self, prob['ks.KS'][1], 34.0)
        assert_rel_error(self, prob['ks.KS'][2], 51.0)

        partials = prob.check_partials(includes=['ks'], out_stream=None)

        for (of, wrt) in partials['ks']:
            assert_rel_error(self, partials['ks'][of, wrt]['abs error'][0], 0.0, 1e-6)

    def test_partials_no_compute(self):
        prob = om.Problem()

        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([5.0, 4.0])))

        ks_comp = model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('px.x', 'ks.g')

        prob.setup()
        prob.run_driver()

        # compute partials with the current model inputs
        inputs = { 'g': prob['ks.g'] }
        partials = {}

        ks_comp.compute_partials(inputs, partials)
        assert_rel_error(self, partials[('KS', 'g')], np.array([1., 0.]), 1e-6)

        # swap inputs and call compute partials again, without calling compute
        inputs['g'][0][0] = 4
        inputs['g'][0][1] = 5

        ks_comp.compute_partials(inputs, partials)
        assert_rel_error(self, partials[('KS', 'g')], np.array([0., 1.]), 1e-6)

    def test_beam_stress(self):
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        max_bending = 100.0

        num_cp = 5
        num_elements = 25
        num_load_cases = 2

        prob = om.Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume, max_bending = max_bending,
                                                    num_elements=num_elements, num_cp=num_cp,
                                                    num_load_cases=num_load_cases))

        prob.driver = om.ScipyOptimizeDriver()
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

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['upper'] = 16.0
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'][0], -1.0)

    def test_lower_flag(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['lower_flag'] = True
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'][0], -12.0)

    def test_deprecated_ks_component(self):
        # run same test as above, only with the deprecated component,
        # to ensure we get the warning and the correct answer.
        # self-contained, to be removed when class name goes away.
        from openmdao.components.ks_comp import KSComponent  # deprecated

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2,))))
        model.add_subsystem('comp', DoubleArrayComp())

        msg = "'KSComponent' has been deprecated. Use 'KSComp' instead."

        with assert_warning(DeprecationWarning, msg):
            model.add_subsystem('ks', KSComponent(width=2))

        model.connect('px.x', 'comp.x1')
        model.connect('comp.y2', 'ks.g')

        model.add_design_var('px.x')
        model.add_objective('comp.y1')
        model.add_constraint('ks.KS', upper=0.0)

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, max(prob['comp.y2']), prob['ks.KS'][0])


class TestKSFunctionFeatures(unittest.TestCase):

    def test_basic(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, )),
                                                y=np.zeros((2, ))))
        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'][0], 15.0)

    def test_vectorized(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([[5.0, 4.0], [10.0, 8.0]])))
        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, 2)),
                                                y=np.zeros((2, 2))))
        model.add_subsystem('ks', om.KSComp(width=2, vec_size=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'][0], 15.0)
        assert_rel_error(self, prob['ks.KS'][1], 30.0)

    def test_upper(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, )),
                                                y=np.zeros((2, ))))
        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['upper'] = 16.0
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'][0], -1.0)

    def test_lower_flag(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([5.0, 4.0])))
        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, )),
                                                y=np.zeros((2, ))))
        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('px.x', 'comp.x')
        model.connect('comp.y', 'ks.g')

        model.ks.options['lower_flag'] = True
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['ks.KS'][0], -12.0)


if __name__ == "__main__":
    unittest.main()
