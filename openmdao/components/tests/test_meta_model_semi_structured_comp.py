"""
Unit tests for the semi-structured metamodel component.
"""
import itertools
import unittest

import numpy as np

import openmdao.api as om
from openmdao.components.tests.test_meta_model_structured_comp import SampleMap
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
from openmdao.utils.testing_utils import force_check_partials


# Data for example used in the docs.

data_x = np.array([
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,

    1.3,
    1.3,
    1.3,
    1.3,
    1.3,

    1.6,
    1.6,
    1.6,
    1.6,
    1.6,

    2.1,
    2.1,
    2.1,
    2.1,
    2.1,

    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,

    2.9,
    2.9,
    2.9,
    2.9,
    2.9,

    3.2,
    3.2,
    3.2,
    3.2,

    3.6,
    3.6,
    3.6,
    3.6,
    3.6,
    3.6,

    4.3,
    4.3,
    4.3,
    4.3,

    4.6,
    4.6,
    4.6,
    4.6,
    4.6,
    4.6,

    4.9,
    4.9,
    4.9,
    4.9,
    4.9,
])

data_y = np.array([
    1.0,
    1.5,
    1.6,
    1.7,
    1.9,

    1.0,
    1.5,
    1.6,
    1.7,
    1.9,

    1.0,
    1.5,
    1.6,
    1.7,
    1.9,

    1.0,
    1.6,
    1.7,
    1.9,
    2.4,

    1.3,
    1.7,
    1.9,
    2.4,
    2.6,
    2.9,

    1.9,
    2.1,
    2.3,
    2.5,
    3.1,

    2.3,
    2.5,
    3.1,
    3.7,

    2.3,
    3.1,
    3.3,
    3.7,
    4.1,
    4.2,

    3.3,
    3.6,
    4.0,
    4.5,

    3.9,
    4.2,
    4.4,
    4.5,
    4.6,
    4.7,

    4.4,
    4.5,
    4.6,
    4.7,
    4.9,
])

data_values = 3.0 + np.sin(data_x*0.2) * np.cos(data_y*0.3)


class TestMetaModelSemiStructured(unittest.TestCase):

    def test_vectorized_linear(self):
        # Test using the model we used for the Structured metamodel.
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=x['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=x['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelSemiStructuredComp(method='slinear', extrapolate=True,
                                              training_data_gradients=True, vec_size=3)

        # Convert to the flat table format.
        grid = np.array(list(itertools.product(*[params[0]['values'],
                                                 params[1]['values'],
                                                 params[2]['values']])))

        j = 0
        for param in params:
            comp.add_input(param['name'], grid[:, j])
            j += 1

        for out in outs:
            comp.add_output(out['name'], training_data=outs[0]['values'].flatten())

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = force_check_partials(prob, method='cs', out_stream=None)
        assert_check_partials(partials, rtol=1e-10)

    def test_vectorized_lagrange2(self):
        # Test using the model we used for the Structured metamodel.
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=x['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=x['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelSemiStructuredComp(method='lagrange2', extrapolate=True,
                                              training_data_gradients=True, vec_size=3)

        # Convert to the flat table format.
        grid = np.array(list(itertools.product(*[params[0]['values'],
                                                 params[1]['values'],
                                                 params[2]['values']])))

        j = 0
        for param in params:
            comp.add_input(param['name'], grid[:, j])
            j += 1

        for out in outs:
            comp.add_output(out['name'], training_data=outs[0]['values'].flatten())

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = force_check_partials(prob, method='cs', out_stream=None)
        assert_check_partials(partials, rtol=1e-10)

    def test_vectorized_lagrange3(self):
        # Test using the model we used for the Structured metamodel.
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=x['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=x['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelSemiStructuredComp(method='lagrange3', extrapolate=True,
                                              training_data_gradients=True, vec_size=3)

        # Convert to the flat table format.
        grid = np.array(list(itertools.product(*[params[0]['values'],
                                                 params[1]['values'],
                                                 params[2]['values']])))

        j = 0
        for param in params:
            comp.add_input(param['name'], grid[:, j])
            j += 1

        for out in outs:
            comp.add_output(out['name'], training_data=outs[0]['values'].flatten())

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = force_check_partials(prob, method='cs', out_stream=None)
        assert_check_partials(partials, rtol=1e-10)

    def test_vectorized_akima(self):
        # Test using the model we used for the Structured metamodel.
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=x['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=x['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelSemiStructuredComp(method='akima', extrapolate=True,
                                              training_data_gradients=True, vec_size=3)

        # Convert to the flat table format.
        grid = np.array(list(itertools.product(*[params[0]['values'],
                                                 params[1]['values'],
                                                 params[2]['values']])))

        j = 0
        for param in params:
            comp.add_input(param['name'], grid[:, j])
            j += 1

        for out in outs:
            comp.add_output(out['name'], training_data=outs[0]['values'].flatten())

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = force_check_partials(prob, method='cs', out_stream=None)
        assert_check_partials(partials, rtol=1e-10)

    def test_error_dim(self):
        x = np.array([1.0, 1.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 1.0, 2.0])
        f = np.array([1.0, 2.0, 3.0])

        comp = om.MetaModelSemiStructuredComp(method='akima')
        comp.add_input('x', x)
        comp.add_input('y', y)
        comp.add_output('f', training_data=f)

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', comp)

        msg = "Size mismatch: training data for 'f' is length 3, but" + \
            f" data for 'x' is length 4."
        with self.assertRaisesRegex(ValueError, msg):
            prob.setup()

    def test_error_no_training_data(self):
        x = np.array([1.0, 1.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 1.0, 2.0])

        comp = om.MetaModelSemiStructuredComp(method='akima')
        comp.add_input('x', x)
        comp.add_input('y', y)

        msg = "Training data is required for output 'f'."
        with self.assertRaisesRegex(ValueError, msg):
            comp.add_output('f')

    def test_list_input(self):
        x = [1.0, 1.0, 2.0, 2.0, 2.0]
        y = [1.0, 2.0, 1.0, 2.0, 3.0]
        f = [1.0, 2.5, 1.5, 4.0, 4.5]

        comp = om.MetaModelSemiStructuredComp(method='slinear', training_data_gradients=True, extrapolate=False)
        comp.add_input('x', x)
        comp.add_input('y', y)
        comp.add_output('f', training_data=f)

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', comp)

        prob.setup()

        prob.set_val('comp.x', 1.5)
        prob.set_val('comp.y', 1.5)

        prob.run_model()

        f = prob.get_val('comp.f')
        assert_near_equal(f, 2.25)

        # Attempt internal extrapolation.
        prob.set_val('comp.x', 1.5)
        prob.set_val('comp.y', 2.5)

        msg = "'comp' <class MetaModelSemiStructuredComp>: Error interpolating output 'f' because input 'comp.y' required extrapolation while interpolating dimension 2, where its value '2.5' exceeded the range ('[1.]', '[2.]')"
        with self.assertRaises(om.AnalysisError) as cm:
            prob.run_model()

        self.assertEqual(str(cm.exception), msg)

    def test_simple(self):

        prob = om.Problem()
        model = prob.model

        interp = om.MetaModelSemiStructuredComp(method='lagrange2', training_data_gradients=True)
        interp.add_input('x', data_x)
        interp.add_input('y', data_y)
        interp.add_output('f', training_data=data_values)

        # Sneak in a multi-output case.
        interp.add_output('g', training_data=2.0 * data_values)

        model.add_subsystem('interp', interp)

        prob.setup(force_alloc_complex=True)

        prob.set_val('interp.x', np.array([3.1]))
        prob.set_val('interp.y', np.array([2.75]))

        prob.run_model()

        assert_near_equal(prob.get_val('interp.f'), 3.39415716, 1e-7)
        assert_near_equal(prob.get_val('interp.g'), 2.0 * 3.39415716, 1e-7)

    def test_simple_training_inputs(self):
        prob = om.Problem()
        model = prob.model

        interp = om.MetaModelSemiStructuredComp(method='lagrange2', training_data_gradients=True)
        interp.add_input('x', data_x)
        interp.add_input('y', data_y)
        interp.add_output('f', training_data=np.zeros(len(data_x)))

        model.add_subsystem('interp', interp)

        prob.setup(force_alloc_complex=True)

        prob.set_val('interp.x', np.array([3.1]))
        prob.set_val('interp.y', np.array([2.75]))

        prob.set_val('interp.f_train', data_values)

        prob.run_model()

        assert_near_equal(prob.get_val('interp.f'), 3.39415716, 1e-7)

    def test_dynamic_training(self):
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        P1 = P1.ravel()
        P2 = P2.ravel()
        P3 = P3.ravel()

        class TableGen(om.ExplicitComponent):

            def setup(self):
                self.add_input('k', 1.0)
                self.add_output('values', np.zeros(len(P1)))

                self.declare_partials('values', 'k')

            def compute(self, inputs, outputs):
                k = inputs['k']

                outputs['values'] = np.sqrt(P1) + P2 * P3 * k

            def compute_partials(self, inputs, partials):
                partials['values', 'k'] = P2 * P3


        prob = om.Problem()
        model = prob.model

        model.add_subsystem('tab', TableGen())

        interp = om.MetaModelSemiStructuredComp(method='lagrange3', training_data_gradients=True)
        interp.add_input('p1', P1)
        interp.add_input('p2', P2)
        interp.add_input('p3', P3)

        interp.add_output('f')

        model.add_subsystem('comp', interp)

        model.connect('tab.values', 'comp.f_train')
        prob.setup(force_alloc_complex=True)

        prob.set_val('comp.p1', 55.12)
        prob.set_val('comp.p2', -2.14)
        prob.set_val('comp.p3', 0.323)

        prob.run_model()

        # we can verify all gradients by checking against finite-difference
        chk = prob.check_totals(of='comp.f', wrt=['tab.k', 'comp.p1', 'comp.p2', 'comp.p3'],
                                   method='cs', out_stream=None);
        assert_check_totals(chk, atol=1e-10, rtol=1e-10)

    def test_detect_local_extrapolation(self):
        # Tests that we detect when any of our points we are using for interpolation are being extrapolated from
        # somewhere else in the semi-structured grid, so that we can adjust our points (if we can.)
        # This test is set up so that if we aren't actively doing this, lagrange2 and lagrange3 will compute
        # large values near the ends. Akima seems to already be robust to this, and didn't require any changes.

        # 8x8 block
        u = np.arange(24)
        v = np.arange(8)

        grid = np.empty((192, 2))
        grid[:, 0] = np.repeat(u, 8)
        grid[:64, 1] = np.tile(v, 8) + 8
        grid[64:128, 1] = np.tile(v, 8)
        grid[128:, 1] = np.tile(v, 8) + 8

        values = np.empty((192, ))
        values[:64] = 1e8 * (6.0 + 5.0 * np.sin(.02 * grid[:64, 0]) + np.sin(.03 * grid[:64, 1]))
        values[64:128] = (6.0 + 5.0 * np.sin(.02 * grid[64:128, 0]) + np.sin(.03 * grid[64:128, 1]))
        values[128:] = 1e8 * (6.0 + 5.0 * np.sin(.02 * grid[128:, 0]) + np.sin(.03 * grid[128:, 1]))

        expected = np.array([6.91181637, 7.01019418, 7.1081943,  7.20577754, 7.30290486, 7.39953742, 7.49563655])

        for method in ['slinear', 'lagrange2', 'lagrange3', 'akima']:

            prob = om.Problem()
            model = prob.model

            interp = om.MetaModelSemiStructuredComp(method=method, vec_size=7)
            interp.add_input('x', grid[:, 0])
            interp.add_input('y', grid[:, 1])
            interp.add_output('f', training_data=values)

            model.add_subsystem('interp', interp)

            prob.setup()

            prob.set_val('interp.x', np.array([8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]))
            prob.set_val('interp.y', np.array([2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2]))

            prob.run_model()

            assert_near_equal(prob.get_val('interp.f'), expected, 1e-3)

    def test_lagrange3_edge_extrapolation_detection_bug(self):
        import itertools

        import numpy as np
        import openmdao.api as om

        grid = np.array(
            [[0, 0],
             [0, 1],
             [0, 2],
             [0, 3],
             [0, 4],
             [1, 0],
             [1, 1],
             [1, 2],
             [1, 3],
             [1, 4],
             [2, 0],
             [2, 1],
             [2, 2],
             [2, 3],
             [2, 4],
             [2, 5],
             [3, 0],
             [3, 1],
             [3, 2],
             [3, 3],
             [3, 4],
             [3, 5],
             [4, 0],
             [4, 1],
             [4, 2],
             [4, 3],
             [4, 4],
             [4, 5]])


        values = 15.0 + 2 * np.random.random(grid.shape[0])

        prob = om.Problem()
        model = prob.model

        interp = om.MetaModelSemiStructuredComp(method='lagrange3')
        interp.add_input('x', training_data=grid[:, 0])
        interp.add_input('y', training_data=grid[:, 1])
        interp.add_output('f', training_data=values)

        model.add_subsystem('interp', interp)

        prob.setup()

        prob.set_val('interp.x', 2.5)
        prob.set_val('interp.y', 4.5)

        # Should run without an Indexerror.
        prob.run_model()


if __name__ == "__main__":
    unittest.main()

