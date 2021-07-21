""" Unit tests for semi structured metamodels in view_mm. """
import unittest

import numpy as np

try:
    import bokeh
    from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization
except ImportError:
    bokeh = None

import openmdao.api as om


@unittest.skipUnless(bokeh, "Bokeh is required")
class SemistructuredMetaModelCompTests(unittest.TestCase):

    def test_basic(self):
        # Tests that semi structured grids load without error.

        grid = np.array([
            [1.0, 5.0, 8.0],
            [1.0, 5.0, 9.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 20.0],
            [1.0, 5.3, 8.0],
            [1.0, 5.3, 9.0],
            [1.0, 5.3, 10.0],
            [1.0, 5.3, 20.0],
            [1.0, 5.6, 8.0],
            [1.0, 5.6, 9.0],
            [1.0, 5.6, 10.0],
            [1.0, 5.6, 20.0],
            [1.0, 6.0, 8.0],
            [1.0, 6.0, 9.0],
            [1.0, 6.0, 10.0],
            [1.0, 6.0, 20.0],
            [2.0, 7.0, 13.0],
            [2.0, 7.0, 14.0],
            [2.0, 7.0, 15.0],
            [2.0, 7.0, 16.0],
            [2.0, 8.0, 13.0],
            [2.0, 8.0, 14.0],
            [2.0, 8.0, 15.0],
            [2.0, 8.0, 16.0],
            [2.0, 8.5, 13.0],
            [2.0, 8.5, 14.0],
            [2.0, 8.5, 15.0],
            [2.0, 8.5, 16.0],
            [2.0, 9.0, 13.0],
            [2.0, 9.0, 14.0],
            [2.0, 9.0, 15.0],
            [2.0, 9.0, 16.0],
        ])

        values = 15.0 + 2 * np.random.random(32)

        prob = om.Problem()
        model = prob.model

        interp = om.MetaModelSemiStructuredComp(vec_size=3, training_data_gradients=True)
        interp.add_input('x', training_data=grid[:, 0])
        interp.add_input('y', training_data=grid[:, 1])
        interp.add_input('z', training_data=grid[:, 2])
        interp.add_output('f', training_data=values)

        model.add_subsystem('interp', interp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        viz = MetaModelVisualization(interp)


if __name__ == '__main__':
    unittest.main()
