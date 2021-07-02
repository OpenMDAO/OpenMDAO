"""
Unit tests for the semi-structured metamodel component.
"""
import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.components.tests.test_meta_model_structured_comp import SampleMap
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_partials


class TestMetaModelStructuredScipy(unittest.TestCase):

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

        for param in params:
            comp.add_input(param['name'])

        # Convert to the flat table format.
        grid = np.array(list(itertools.product(*[params[0]['values'],
                                                 params[1]['values'],
                                                 params[2]['values']])))

        for out in outs:
            comp.add_output(out['name'], grid_points=grid, training_data=outs[0]['values'].flatten())

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, rtol=1e-10)

    def test_vectorized_lagrange(self):
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


        for param in params:
            comp.add_input(param['name'])

        # Convert to the flat table format.
        grid = np.array(list(itertools.product(*[params[0]['values'],
                                                 params[1]['values'],
                                                 params[2]['values']])))

        for out in outs:
            comp.add_output(out['name'], grid_points=grid, training_data=outs[0]['values'].flatten())

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()

