""" Test the KSFunction component. """
import unittest

import numpy as np

try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
except ImportError:
    plt = None

import openmdao.api as om
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup
from openmdao.utils.assert_utils import assert_near_equal


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

        assert_near_equal(max(prob['comp.y2']), prob['ks.KS'][0])

    def test_bad_units(self):
        with self.assertRaises(ValueError) as ctx:
            om.KSComp(units='wtfu')
        self.assertEqual(str(ctx.exception), "The units 'wtfu' are invalid.")

    def test_vectorized(self):
        prob = om.Problem()
        model = prob.model

        x = np.zeros((3, 5))
        x[0, :] = np.array([3.0, 5.0, 11.0, 13.0, 17.0])
        x[1, :] = np.array([13.0, 11.0, 5.0, 17.0, 3.0])*2
        x[2, :] = np.array([11.0, 3.0, 17.0, 5.0, 13.0])*3

        model.add_subsystem('ks', om.KSComp(width=5, vec_size=3))

        model.add_design_var('ks.g')
        model.add_constraint('ks.KS', upper=0.0)

        prob.setup()
        prob.set_val('ks.g', x)
        prob.run_driver()


        assert_near_equal(prob.get_val('ks.KS', indices=0), 17.0)
        assert_near_equal(prob.get_val('ks.KS', indices=1), 34.0)
        assert_near_equal(prob.get_val('ks.KS', indices=2), 51.0)

        prob.model.ks._no_check_partials = False  # override skipping of check_partials

        partials = prob.check_partials(includes=['ks'], out_stream=None)

        for (of, wrt) in partials['ks']:
            assert_near_equal(partials['ks'][of, wrt]['abs error'][0], 0.0, 1e-6)

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
        assert_near_equal(partials[('KS', 'g')], np.array([1., 0.]), 1e-6)

        # swap inputs and call compute partials again, without calling compute
        inputs['g'][0][0] = 4
        inputs['g'][0][1] = 5

        ks_comp.compute_partials(inputs, partials)
        assert_near_equal(partials[('KS', 'g')], np.array([0., 1.]), 1e-6)

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

        # Test that the maximum constraint prior to aggregation is close to "active".
        assert_near_equal(max(stress0), 100.0, tolerance=5e-2)
        assert_near_equal(max(stress1), 100.0, tolerance=5e-2)

        # Test that no original constraint is violated.
        self.assertTrue(np.all(stress0 < 100.0))
        self.assertTrue(np.all(stress1 < 100.0))

    def test_beam_stress_ks_add_constraint(self):
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
                                                    num_load_cases=num_load_cases,
                                                    ks_add_constraint=True))

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        prob.setup(mode='rev')

        prob.run_driver()

        stress0 = prob['parallel.sub_0.stress_comp.stress_0']
        stress1 = prob['parallel.sub_0.stress_comp.stress_1']

        # Test that the the maximum constraint prior to aggregation is close to "active".
        assert_near_equal(max(stress0), 100.0, tolerance=5e-2)
        assert_near_equal(max(stress1), 100.0, tolerance=5e-2)

        # Test that no original constraint is violated.
        self.assertTrue(np.all(stress0 < 100.0))
        self.assertTrue(np.all(stress1 < 100.0))

    def test_upper(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))
        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('comp.y', 'ks.g')

        model.ks.options['upper'] = 16.0
        prob.setup()
        prob.set_val('comp.x', np.array([5.0, 4.0]))
        prob.run_model()

        assert_near_equal(prob.get_val('ks.KS', indices=0), -1.0)

    def test_lower_flag(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y = 3.0*x', x=np.zeros((2, )), y=np.zeros((2, ))))

        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('comp.y', 'ks.g')

        model.ks.options['lower_flag'] = True
        prob.setup()
        prob.set_val('comp.x', np.array([5.0, 4.0]))
        prob.run_model()

        assert_near_equal(prob.get_val('ks.KS', indices=0), -12.0)


class TestKSFunctionFeatures(unittest.TestCase):

    def test_basic(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y = 3.0*x',
                                                x=np.zeros((2, )),
                                                y=np.zeros((2, ))), promotes_inputs=['x'])

        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('comp.y', 'ks.g')

        prob.setup()
        prob.set_val('x', np.array([5.0, 4.0]))
        prob.run_model()

        assert_near_equal(prob.get_val('ks.KS'), [[15.0]])

    def test_vectorized(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y = 3.0*x',
                                                x=np.zeros((2, 2)),
                                                y=np.zeros((2, 2))), promotes_inputs=['x'])
        model.add_subsystem('ks', om.KSComp(width=2, vec_size=2))

        model.connect('comp.y', 'ks.g')

        prob.setup()
        prob.set_val('x', np.array([[5.0, 4.0], [10.0, 8.0]]))
        prob.run_model()

        assert_near_equal(prob.get_val('ks.KS'), np.array([[15], [30]]))

    def test_upper(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y = 3.0*x',
                                                x=np.zeros((2, )),
                                                y=np.zeros((2, ))), promotes_inputs=['x'])
        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('comp.y', 'ks.g')

        model.ks.options['upper'] = 16.0
        prob.setup()
        prob.set_val('x', np.array([5.0, 4.0]))
        prob.run_model()

        assert_near_equal(prob['ks.KS'], np.array([[-1.0]]))

    def test_lower_flag(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y = 3.0*x',
                                                x=np.zeros((2, )),
                                                y=np.zeros((2, ))), promotes_inputs=['x'])

        model.add_subsystem('ks', om.KSComp(width=2))

        model.connect('comp.y', 'ks.g')

        model.ks.options['lower_flag'] = True
        prob.setup()
        prob.set_val('x', np.array([5.0, 4.0]))
        prob.run_model()

        assert_near_equal(prob.get_val('ks.KS'), [[-12.0]])

    @unittest.skipIf(not plt, "requires matplotlib")
    def test_add_constraint(self):
        import numpy as np
        import openmdao.api as om
        import matplotlib.pyplot as plt

        n = 50
        prob = om.Problem()
        model = prob.model

        prob.driver = om.ScipyOptimizeDriver()

        model.add_subsystem('comp', om.ExecComp('y = -3.0*x**2 + k',
                                                x=np.zeros((n, )),
                                                y=np.zeros((n, )),
                                                k=0.0), promotes_inputs=['x', 'k'])

        model.add_subsystem('ks', om.KSComp(width=n, upper=4.0, add_constraint=True))

        model.add_design_var('k', lower=-10, upper=10)
        model.add_objective('k', scaler=-1)

        model.connect('comp.y', 'ks.g')

        prob.setup()
        prob.set_val('x', np.linspace(-np.pi/2, np.pi/2, n))
        prob.set_val('k', 5.)

        prob.run_driver()

        self.assertTrue(max(prob.get_val('comp.y')) <= 4.0)

        fig, ax = plt.subplots()

        x = prob.get_val('x')
        y = prob.get_val('comp.y')

        ax.plot(x, y, 'r.')
        ax.plot(x, 4.0*np.ones_like(x), 'k--')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.text(-0.25, 0, f"k = {prob.get_val('k')[0]:6.3f}")

        plt.show()

    def test_units(self):
        import openmdao.api as om
        from openmdao.utils.units import convert_units

        n = 10

        model = om.Group()

        model.add_subsystem('ks', om.KSComp(width=n, units='m'), promotes_inputs=[('g', 'x')])
        model.set_input_defaults('x', range(n), units='ft')

        prob = om.Problem(model=model)
        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('ks.KS', indices=0), np.amax(prob.get_val('x')), tolerance=1e-8)


if __name__ == "__main__":
    unittest.main()
