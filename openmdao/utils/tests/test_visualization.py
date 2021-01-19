import unittest

import numpy as np

try:
    import matplotlib
except ImportError:
    matplotlib = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

@unittest.skipUnless(matplotlib, "Matplotlib is required.")
class TestVisualization(unittest.TestCase):

    def setUp(self):
        matplotlib.use('Agg') # so no plots are actually drawn

    def test_partial_deriv_plot(self):

        class ArrayComp2D(om.ExplicitComponent):
            """
            A fairly simple array component with an intentional error in compute_partials.
            """
            def setup(self):
                self.JJ = np.array([[1.0, 0.0, 0.0, 7.0],
                                    [0.0, 2.5, 0.0, 0.0],
                                    [-1.0, 0.0, 8.0, 0.0],
                                    [0.0, 4.0, 0.0, 6.0]])
                # Params
                self.add_input('x1', np.zeros([4]))
                # Unknowns
                self.add_output('y1', np.zeros([4]))
                self.declare_partials(of='*', wrt='*')
            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])
            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                # create some error to force the diff plot to show something
                error = np.zeros((4,4))
                err = 1e-7
                error[0][3] = err
                error[1][2] = - 2.0 * err
                partials[('y1', 'x1')] = self.JJ + error

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))), promotes=['x1'])
        model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])
        prob.setup(check=False, mode='fwd')
        check_partials_data = prob.check_partials(out_stream=None)

        # plot with defaults
        fig, ax = om.partial_deriv_plot('y1', 'x1', check_partials_data, title="Defaults")
        # Instead of seeing if the images created by matplotlib match what we expect, which
        # is a fragile thing to do in testing, check a data structure inside matplotlib's
        # objects. We will assume matplotlib draws the correct thing.
        expected_array = np.array([[1., 0., 0., 1.],
                                   [0., 1., 0., 0.],
                                   [1., 0., 1., 0.],
                                   [0., 1., 0., 1.]])
        actual_array = ax[0].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)
        expected_array = np.array([[ 1.,  0.,  0.,  1.],
                                   [ 0.,  1.,  1.,  0.],
                                   [ 1.,  0.,  1.,  0.],
                                   [ 0.,  1.,  0.,  1.]])
        actual_array = ax[1].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)
        expected_array = np.array([[  9.31322575e-10,   0.0,   0.0, 1.0e-07],
                                   [  0.0,   0.0,  -2.0e-07, 0.0],
                                   [  0.0,   0.0,   9.31322575e-10, 0.0],
                                   [  0.0,   0.0,   0.0, 1.86264515e-09]])
        actual_array = ax[2].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)

        # plot with specified jac_method
        fig, ax = om.partial_deriv_plot('y1', 'x1', check_partials_data, jac_method = "J_fwd",
                                        title="specified jac_method")
        expected_array = np.array([[1., 0., 0., 1.],
                                   [0., 1., 0., 0.],
                                   [1., 0., 1., 0.],
                                   [0., 1., 0., 1.]])
        actual_array = ax[0].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)
        expected_array = np.array([[ 1.,  0.,  0.,  1.],
                                   [ 0.,  1.,  1.,  0.],
                                   [ 1.,  0.,  1.,  0.],
                                   [ 0.,  1.,  0.,  1.]])
        actual_array = ax[1].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)
        expected_array = np.array([[  9.31322575e-10,   0.0,   0.0, 1.0e-07],
                                   [  0.0,   0.0,  -2.0e-07, 0.0],
                                   [  0.0,   0.0,   9.31322575e-10, 0.0],
                                   [  0.0,   0.0,   0.0, 1.86264515e-09]])
        actual_array = ax[2].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)

        # plot in non-binary mode
        fig, ax = om.partial_deriv_plot('y1', 'x1', check_partials_data, binary = False,
                                        title="non-binary")
        expected_array = np.array([[ 1. ,  0. ,  0. ,  7. ],
                                   [ 0. ,  2.5,  0. ,  0. ],
                                   [-1. ,  0. ,  8. ,  0. ],
                                   [ 0. ,  4. ,  0. ,  6. ]])
        actual_array = ax[0].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)
        expected_array = np.array([[  1.0e+00,   0.0,   0.0, 7.00000010e+00],
                                   [  0.0,   2.5e+00,  -2.0e-07, 0.0],
                                   [ -1.0e+00,   0.0,   8.0e+00, 0.0],
                                   [  0.0,   4.0e+00,   0.0, 6.0e+00]])
        actual_array = ax[1].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)
        expected_array = np.array([[  9.31322575e-10,   0.0,   0.0, 1.0e-07],
                                   [  0.0,   0.0,  -2.0e-07, 0.0],
                                   [  0.0,   0.0,   9.31322575e-10, 0.0],
                                   [  0.0,   0.0,   0.0, 1.86264515e-09]])
        actual_array = ax[2].images[0]._A.data
        assert_near_equal(expected_array, actual_array, 1e-8)

        # plot with different tol values
        # Not obvious how to test this other than image matching
        om.partial_deriv_plot('y1', 'x1', check_partials_data, tol=1e-5,
                              title="tol greater than err")
        om.partial_deriv_plot('y1', 'x1', check_partials_data, tol=1e-10,
                              title="tol less than err")


@unittest.skipUnless(matplotlib, "Matplotlib is required.")
class TestFeatureVisualization(unittest.TestCase):

    def setUp(self):
        matplotlib.use('Agg') # so no plots are actually drawn in interactive mode

    def test_partial_deriv_plot(self):
        import numpy as np
        import openmdao.api as om

        class ArrayComp2D(om.ExplicitComponent):
            """
            A fairly simple array component with an intentional error in compute_partials.
            """
            def setup(self):
                self.JJ = np.array([[1.0, 0.0, 0.0, 7.0],
                                    [0.0, 2.5, 0.0, 0.0],
                                    [-1.0, 0.0, 8.0, 0.0],
                                    [0.0, 4.0, 0.0, 6.0]])
                # Params
                self.add_input('x1', np.ones([4]))
                # Unknowns
                self.add_output('y1', np.zeros([4]))
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                # create some error to force the diff plot to show something
                error = np.zeros((4, 4))
                err = 1e-7
                error[0][3] = err
                error[1][2] = - 2.0 * err
                partials[('y1', 'x1')] = self.JJ + error

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])
        prob.setup(check=False, mode='fwd')
        check_partials_data = prob.check_partials(out_stream=None)

        # plot with defaults
        om.partial_deriv_plot('y1', 'x1', check_partials_data)

    def test_partial_deriv_non_binary_plot(self):
        import numpy as np
        import openmdao.api as om

        class ArrayComp2D(om.ExplicitComponent):
            """
            A fairly simple array component with an intentional error in compute_partials.
            """
            def setup(self):
                self.JJ = np.array([[1.0, 0.0, 0.0, 7.0],
                                    [0.0, 2.5, 0.0, 0.0],
                                    [-1.0, 0.0, 8.0, 0.0],
                                    [0.0, 4.0, 0.0, 6.0]])
                # Params
                self.add_input('x1', np.ones([4]))
                # Unknowns
                self.add_output('y1', np.zeros([4]))
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                # create some error to force the diff plot to show something
                error = np.zeros((4, 4))
                err = 1e-7
                error[0][3] = err
                error[1][2] = - 2.0 * err
                partials[('y1', 'x1')] = self.JJ + error

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])
        prob.setup(check=False, mode='fwd')
        check_partials_data = prob.check_partials(out_stream=None)

        # plot in non-binary mode
        om.partial_deriv_plot('y1', 'x1', check_partials_data, binary = False)


if __name__ == "__main__":

    unittest.main()
