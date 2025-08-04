"""
Test Sampling Generators.
"""
import unittest

import numpy as np

import openmdao.api as om

from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.drivers.sampling.uniform_generator import UniformGenerator


class ParaboloidArray(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + x*y + (y+4)^2 - 3.

    Where x and y are xy[0] and xy[1] respectively.
    """

    def setup(self):
        self.add_input('xy', val=np.array([0., 0.]))
        self.add_output('f_xy', val=0.0)

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """
        x = inputs['xy'][0]
        y = inputs['xy'][1]
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0


@use_tempdirs
class TestUniformGenerator(unittest.TestCase):

    def test_uniform(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)

        factors = {
            'x': {'lower': -10, 'upper': 10},
            'y': {'lower': -10, 'upper': 10},
        }

        prob.driver = om.AnalysisDriver(UniformGenerator(factors, num_samples=5, seed=0))
        prob.driver.add_response('f_xy')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # all values should be between -10 and 10, check expected values for seed = 0
        expected = [
            {'x': np.array([0.97627008]), 'y': np.array([4.30378733])},
            {'x': np.array([2.05526752]), 'y': np.array([0.89766366])},
            {'x': np.array([-1.52690401]), 'y': np.array([2.91788226])},
            {'x': np.array([-1.24825577]), 'y': np.array([7.83546002])},
            {'x': np.array([9.27325521]), 'y': np.array([-2.33116962])},
        ]

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 5)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y'):
                assert_near_equal(outputs[name], expected_case[name], 1e-4)


if __name__ == "__main__":
    unittest.main()
