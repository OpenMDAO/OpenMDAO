""" Unit tests for structured metamodels in view_mm. """
import unittest
import subprocess
import os

import numpy as np
from numpy.testing import assert_almost_equal

try:
    import bokeh
    from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization
except ImportError:
    bokeh = None

import openmdao.api as om
import openmdao.test_suite.test_examples.meta_model_examples.structured_meta_model_example as example

@unittest.skipUnless(bokeh, "Bokeh is required")
class StructuredMetaModelCompTests(unittest.TestCase):

    csv_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'known_data_point_files')

    def setUp(self):
        self.grid_data = np.array([[308.12909601, 253.61567418, 204.6578079, 161.25549718, 123.40874201, 91.1175424,   64.38189835,  43.20180985,  27.5772769,   17.50829952],
                                  [162.89542418, 123.20470795,  89.06954726,  60.48994214,  37.46589257, 19.99739855,   8.08446009,   1.72707719,   0.92524984,   5.67897804,],
                                  [ 90.2866907,   63.02637433,  41.32161352,  25.17240826,  14.57875856, 9.54066442,  10.05812583,  16.13114279,  27.75971531,  44.94384339,],
                                  [ 55.60211264,  38.37989042,  26.71322375,  20.60211264,  20.04655709, 25.04655709,  35.60211264,  51.71322375,  73.37989042, 100.60211264],
                                  [ 22.81724065,  13.24080685,   9.2199286,   10.75460591,  17.84483877, 30.49062719, 48.69197117,  72.4488707,  101.76132579, 136.62933643],
                                  [  5.11168719,   0.78873608,   2.02134053,   8.80950054,  21.1532161, 39.05248721,  62.50731389,  91.51769611, 126.0836339,  166.20512723],
                                  [ 14.3413983,   12.87962416,  16.97340558,  26.62274256,  41.82763509, 62.58808317,  88.90408682, 120.77564601, 158.20276077, 201.18543108],
                                  [ 20.18431209,  19.1914092,   23.75406186,  33.87227009,  49.54603386, 70.77535319,  97.56022808, 129.90065853, 167.79664453, 211.24818608],
                                  [  8.48953212,   5.57319475,   8.21241294,  16.40718668,  30.15751598, 49.46340083,  74.32484124, 104.74183721, 140.71438873, 182.2424958 ],
                                  [ 10.96088904,   3.72881146,   2.05228945,   5.93132298,  15.36591208, 30.35605673,  50.90175693,  77.00301269, 108.65982401, 145.87219088]])

        num_train = 10

        x0_min, x0_max = -5.0, 10.0
        x1_min, x1_max = 0.0, 15.0
        train_x0 = np.linspace(x0_min, x0_max, num_train)
        train_x1 = np.linspace(x1_min, x1_max, num_train)
        t_data = self.grid_data

        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output('x0', 0.0)
        ivc.add_output('x1', 0.0)

        prob.model.add_subsystem('p', ivc, promotes=['*'])
        self.mm = mm = prob.model.add_subsystem('mm', om.MetaModelStructuredComp(method='slinear'),
                                    promotes=['x0', 'x1'])
        mm.add_input('x0', 0.0, train_x0)
        mm.add_input('x1', 0.0, train_x1)
        mm.add_output('f', 0.0, t_data)

        prob.setup()
        prob.final_setup()

    def test_working_scipy_slinear(self):

        # Create regular grid interpolator instance
        xor_interp = om.MetaModelStructuredComp(method='scipy_slinear')

        # set up inputs and outputs
        xor_interp.add_input('x', 0.0, training_data=np.array([0.0, 1.0]), units=None)
        xor_interp.add_input('y', 1.0, training_data=np.array([0.0, 1.0]), units=None)

        xor_interp.add_output(
            'xor', 1.0, training_data=np.array([[0.0, 1.0], [1.0, 0.0]]), units=None)

        # Set up the OpenMDAO model
        model = om.Group()
        ivc = om.IndepVarComp()
        ivc.add_output('x', 0.0)
        ivc.add_output('y', 1.0)
        model.add_subsystem('ivc', ivc, promotes=["*"])
        model.add_subsystem('comp', xor_interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        MetaModelVisualization(xor_interp)

    def test_working_slinear(self):

        num_train = 10

        x0_min, x0_max = -5.0, 10.0
        x1_min, x1_max = 0.0, 15.0
        train_x0 = np.linspace(x0_min, x0_max, num_train)
        train_x1 = np.linspace(x1_min, x1_max, num_train)
        t_data = self.grid_data

        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output('x0', 0.0)
        ivc.add_output('x1', 0.0)

        prob.model.add_subsystem('p', ivc, promotes=['*'])
        mm = prob.model.add_subsystem('mm', om.MetaModelStructuredComp(method='slinear'),
                                    promotes=['x0', 'x1'])
        mm.add_input('x0', 0.0, train_x0)
        mm.add_input('x1', 0.0, train_x1)
        mm.add_output('f', 0.0, t_data)

        prob.setup()
        prob.final_setup()

        MetaModelVisualization(mm)


    def test_working_scipy_cubic(self):

        # create input param training data, of sizes 25, 5, and 10 points resp.
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        # Create regular grid interpolator instance
        interp = om.MetaModelStructuredComp(method='scipy_cubic')
        interp.add_input('p1', 0.5, training_data=p1)
        interp.add_input('p2', 0.0, training_data=p2)
        interp.add_input('p3', 3.14, training_data=p3)

        interp.add_output('f', 0.0, training_data=f)

        # Set up the OpenMDAO model
        model = om.Group()
        model.add_subsystem('comp', interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        MetaModelVisualization(interp)

    def test_working_cubic(self):

        # create input param training data, of sizes 25, 5, and 10 points resp.
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        # Create regular grid interpolator instance
        interp = om.MetaModelStructuredComp(method='cubic')
        interp.add_input('p1', 0.5, training_data=p1)
        interp.add_input('p2', 0.0, training_data=p2)
        interp.add_input('p3', 3.14, training_data=p3)

        interp.add_output('f', 0.0, training_data=f)

        # Set up the OpenMDAO model
        model = om.Group()
        model.add_subsystem('comp', interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        MetaModelVisualization(interp)

    def test_working_multipoint_scipy_cubic(self):

        # create input param training data, of sizes 25, 5, and 10 points resp.
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        # Create regular grid interpolator instance
        interp = om.MetaModelStructuredComp(method='scipy_cubic', vec_size=2)
        interp.add_input('p1', 0.5, training_data=p1)
        interp.add_input('p2', 0.0, training_data=p2)
        interp.add_input('p3', 3.14, training_data=p3)

        interp.add_output('f', 0.0, training_data=f)

        # Set up the OpenMDAO model
        model = om.Group()
        model.add_subsystem('comp', interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        MetaModelVisualization(interp)

    def test_aligned_training_points(self):

        known_points_right = np.array([[ 10.        ,   0.        ,   0.        ,  10.96088904],
                                [ 10.        ,   1.66666667,   0.        ,   3.72881146],
                                [ 10.        ,   3.33333333,   0.        ,   2.05228945],
                                [ 10.        ,   5.        ,   0.        ,   5.93132298],
                                [ 10.        ,   6.66666667,   0.        ,  15.36591208],
                                [ 10.        ,   8.33333333,   0.        ,  30.35605673],
                                [ 10.        ,  10.        ,   0.        ,  50.90175693],
                                [ 10.        ,  11.66666667,   0.        ,  77.00301269],
                                [ 10.        ,  13.33333333,   0.        , 108.65982401],
                                [ 10.        ,  15.        ,   0.        , 145.87219088]])

        known_points_bottom = np.array([[ -5.        ,  15.        ,   0.        ,  17.50829952],
                                [ -3.33333333,  15.        ,   0.        ,   5.67897804],
                                [ -1.66666667,  15.        ,   0.        ,  44.94384339],
                                [  0.        ,  15.        ,   0.        , 100.60211264],
                                [  1.66666667,  15.        ,   0.        , 136.62933643],
                                [  3.33333333,  15.        ,   0.        , 166.20512723],
                                [  5.        ,  15.        ,   0.        , 201.18543108],
                                [  6.66666667,  15.        ,   0.        , 211.24818608],
                                [  8.33333333,  15.        ,   0.        , 182.2424958 ],
                                [ 10.        ,  15.        ,   0.        , 145.87219088]])

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [10, 15]
        right_points = adjusted_points._structured_training_points(compute_distance=True, source='right')
        bottom_points = adjusted_points._structured_training_points(compute_distance=True, source='bottom')

        # Make sure that arrays equal each other to the 8th decimal place
        assert_almost_equal(known_points_right, right_points, decimal=8)
        assert_almost_equal(known_points_bottom, bottom_points, decimal=8)

    def test_in_between_training_points_right(self):

        known_points = np.array([[0.00000000e+00, 0.00000000e+00, 5.44217687e-02, 5.56021126e+01],
                                [0.00000000e+00, 1.66666667e+00, 5.44217687e-02, 3.83798904e+01],
                                [0.00000000e+00, 3.33333333e+00, 5.44217687e-02, 2.67132238e+01],
                                [0.00000000e+00, 5.00000000e+00, 5.44217687e-02, 2.06021126e+01],
                                [0.00000000e+00, 6.66666667e+00, 5.44217687e-02, 2.00465571e+01],
                                [0.00000000e+00, 8.33333333e+00, 5.44217687e-02, 2.50465571e+01],
                                [0.00000000e+00, 1.00000000e+01, 5.44217687e-02, 3.56021126e+01],
                                [0.00000000e+00, 1.16666667e+01, 5.44217687e-02, 5.17132237e+01],
                                [0.00000000e+00, 1.33333333e+01, 5.44217687e-02, 7.33798904e+01],
                                [0.00000000e+00, 1.50000000e+01, 5.44217687e-02, 1.00602113e+02],
                                [1.66666667e+00, 0.00000000e+00, 5.66893424e-02, 2.28172406e+01],
                                [1.66666667e+00, 1.66666667e+00, 5.66893424e-02, 1.32408069e+01],
                                [1.66666667e+00, 3.33333333e+00, 5.66893424e-02, 9.21992860e+00],
                                [1.66666667e+00, 5.00000000e+00, 5.66893424e-02, 1.07546059e+01],
                                [1.66666667e+00, 6.66666667e+00, 5.66893424e-02, 1.78448388e+01],
                                [1.66666667e+00, 8.33333333e+00, 5.66893424e-02, 3.04906272e+01],
                                [1.66666667e+00, 1.00000000e+01, 5.66893424e-02, 4.86919712e+01],
                                [1.66666667e+00, 1.16666667e+01, 5.66893424e-02, 7.24488707e+01],
                                [1.66666667e+00, 1.33333333e+01, 5.66893424e-02, 1.01761326e+02],
                                [1.66666667e+00, 1.50000000e+01, 5.66893424e-02, 1.36629336e+02]])

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [0.8163265306122387, 0.0]
        new_points = adjusted_points._structured_training_points(compute_distance=True, source='right')

        assert_almost_equal(known_points, new_points, decimal=5)

    def test_in_between_training_points_bottom(self):

        known_points = np.array([[-5.00000000e+00,  6.66666667e+00,  6.57596372e-02, 1.23408742e+02],
                                [-5.00000000e+00,  8.33333333e+00,  4.53514739e-02, 9.11175424e+01],
                                [-3.33333333e+00,  6.66666667e+00,  6.57596372e-02, 3.74658926e+01],
                                [-3.33333333e+00,  8.33333333e+00,  4.53514739e-02, 1.99973985e+01],
                                [-1.66666667e+00,  6.66666667e+00,  6.57596372e-02, 1.45787586e+01],
                                [-1.66666667e+00,  8.33333333e+00,  4.53514739e-02, 9.54066442e+00],
                                [ 0.00000000e+00,  6.66666667e+00,  6.57596372e-02, 2.00465571e+01],
                                [ 0.00000000e+00,  8.33333333e+00,  4.53514739e-02, 2.50465571e+01],
                                [ 1.66666667e+00,  6.66666667e+00,  6.57596372e-02, 1.78448388e+01],
                                [ 1.66666667e+00,  8.33333333e+00,  4.53514739e-02, 3.04906272e+01],
                                [ 3.33333333e+00,  6.66666667e+00,  6.57596372e-02, 2.11532161e+01],
                                [ 3.33333333e+00,  8.33333333e+00,  4.53514739e-02, 3.90524872e+01],
                                [ 5.00000000e+00,  6.66666667e+00,  6.57596372e-02, 4.18276351e+01],
                                [ 5.00000000e+00,  8.33333333e+00,  4.53514739e-02, 6.25880832e+01],
                                [ 6.66666667e+00,  6.66666667e+00,  6.57596372e-02, 4.95460339e+01],
                                [ 6.66666667e+00,  8.33333333e+00,  4.53514739e-02, 7.07753532e+01],
                                [ 8.33333333e+00,  6.66666667e+00,  6.57596372e-02, 3.01575160e+01],
                                [ 8.33333333e+00,  8.33333333e+00,  4.53514739e-02, 4.94634008e+01],
                                [ 1.00000000e+01,  6.66666667e+00,  6.57596372e-02, 1.53659121e+01],
                                [ 1.00000000e+01,  8.33333333e+00,  4.53514739e-02, 3.03560567e+01]])

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [-5, 7.653061224489797]
        new_points = adjusted_points._structured_training_points(compute_distance=True, source='bottom')

        assert_almost_equal(known_points, new_points, decimal=5)

    def test_flip_inputs_aligned_points(self):

        known_points_right = np.array([[6.66666667e+00, 0.00000000e+00, 2.26757370e-03, 2.01843121e+01],
                                      [6.66666667e+00, 1.66666667e+00, 2.26757370e-03, 1.91914092e+01],
                                      [6.66666667e+00, 3.33333333e+00, 2.26757370e-03, 2.37540619e+01],
                                      [6.66666667e+00, 5.00000000e+00, 2.26757370e-03, 3.38722701e+01],
                                      [6.66666667e+00, 6.66666667e+00, 2.26757370e-03, 4.95460339e+01],
                                      [6.66666667e+00, 8.33333333e+00, 2.26757370e-03, 7.07753532e+01],
                                      [6.66666667e+00, 1.00000000e+01, 2.26757370e-03, 9.75602281e+01],
                                      [6.66666667e+00, 1.16666667e+01, 2.26757370e-03, 1.29900659e+02],
                                      [6.66666667e+00, 1.33333333e+01, 2.26757370e-03, 1.67796645e+02],
                                      [6.66666667e+00, 1.50000000e+01, 2.26757370e-03, 2.11248186e+02]])

        known_points_bottom = np.array([[-5.00000000e+00,  3.33333333e+00,  2.26757370e-03, 2.04657808e+02],
                                        [-3.33333333e+00,  3.33333333e+00,  2.26757370e-03, 8.90695473e+01],
                                        [-1.66666667e+00,  3.33333333e+00,  2.26757370e-03, 4.13216135e+01],
                                        [ 0.00000000e+00,  3.33333333e+00,  2.26757370e-03, 2.67132238e+01],
                                        [ 1.66666667e+00,  3.33333333e+00,  2.26757370e-03, 9.21992860e+00],
                                        [ 3.33333333e+00,  3.33333333e+00,  2.26757370e-03, 2.02134053e+00],
                                        [ 5.00000000e+00,  3.33333333e+00,  2.26757370e-03, 1.69734056e+01],
                                        [ 6.66666667e+00,  3.33333333e+00,  2.26757370e-03, 2.37540619e+01],
                                        [ 8.33333333e+00,  3.33333333e+00,  2.26757370e-03, 8.21241294e+00],
                                        [ 1.00000000e+01,  3.33333333e+00,  2.26757370e-03, 2.05228945e+00]])

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [6.632653061224477, 3.36734693877551]
        right_points = adjusted_points._structured_training_points(compute_distance=True, source='right')
        bottom_points = adjusted_points._structured_training_points(compute_distance=True, source='bottom')


        assert_almost_equal(known_points_right, right_points, decimal=5)
        assert_almost_equal(known_points_bottom, bottom_points, decimal=5)

    def test_updated_scatter_distance(self):

        filename = os.path.join(self.csv_dir, 'updated_scatter_distance.csv')

        known_points_bottom = np.genfromtxt(
            filename, delimiter=',', usecols=(5,6,7,8))
        known_points_right = np.genfromtxt(
            filename, delimiter=',', usecols=(0,1,2,3))

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [6.632653061224477, 3.36734693877551]
        adjusted_points.dist_range = 0.5

        right_points = adjusted_points._structured_training_points(compute_distance=True, source='right')
        bottom_points = adjusted_points._structured_training_points(compute_distance=True, source='bottom')


        assert_almost_equal(known_points_right, right_points, decimal=5)
        assert_almost_equal(known_points_bottom, bottom_points, decimal=5)

    def test_five_alpha_points(self):
        filename = os.path.join(self.csv_dir, 'test_five_alpha_points.csv')

        known_points_bottom = np.genfromtxt(
            filename, delimiter=',', usecols=(1))
        known_points_right = np.genfromtxt(
            filename, delimiter=',', usecols=(0))

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [6.632653061224477, 3.36734693877551]
        adjusted_points.dist_range = 0.5

        right_points = adjusted_points._structured_training_points(compute_distance=True, source='right')
        right_plot = adjusted_points._right_plot()
        right_transparency = adjusted_points.right_alphas

        bottom_points = adjusted_points._structured_training_points(compute_distance=True, source='bottom')
        bottom_plot = adjusted_points._bottom_plot()
        bottom_transparency = adjusted_points.bottom_alphas


        assert_almost_equal(known_points_right, right_transparency, decimal=5)
        assert_almost_equal(known_points_bottom, bottom_transparency, decimal=5)

    def test_single_line_of_alpha_points(self):

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [6.632653061224477, 3.36734693877551]

        right_points = adjusted_points._structured_training_points(compute_distance=True, source='right')
        right_plot = adjusted_points._right_plot()

        bottom_points = adjusted_points._structured_training_points(compute_distance=True, source='bottom')
        bottom_plot = adjusted_points._bottom_plot()


        self.assertTrue(len(adjusted_points.right_alphas) == 10)
        self.assertTrue(len(adjusted_points.bottom_alphas) == 10)

if __name__ == '__main__':
    unittest.main()
