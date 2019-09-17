import unittest

import numpy as np
import openmdao.api as om
from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization


class StructuredMetaModelCompTests(unittest.TestCase):

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
        # Create regular grid interpolator instance
        xor_interp = om.MetaModelStructuredComp(method='slinear')

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

    # def test_working_akima(self):

    #     # create input param training data, of sizes 25, 5, and 10 points resp.
    #     p1 = np.linspace(0, 100, 25)
    #     p2 = np.linspace(-10, 10, 5)
    #     p3 = np.linspace(0, 1, 10)

    #     # can use meshgrid to create a 3D array of test data
    #     P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
    #     f = np.sqrt(P1) + P2 * P3

    #     # Create regular grid interpolator instance
    #     interp = om.MetaModelStructuredComp(method='akima', vec_size=2)
    #     interp.add_input('p1', 0.5, training_data=p1)
    #     interp.add_input('p2', 0.0, training_data=p2)
    #     interp.add_input('p3', 3.14, training_data=p3)

    #     interp.add_output('f', 0.0, training_data=f)

    #     # Set up the OpenMDAO model
    #     model = om.Group()
    #     model.add_subsystem('comp', interp, promotes=["*"])
    #     prob = om.Problem(model)
    #     prob.setup()

    #     MetaModelVisualization(interp)

    # def test_working_scipy_quintic(self):

    #     # create input param training data, of sizes 25, 5, and 10 points resp.
    #     p1 = np.linspace(0, 100, 25)
    #     p2 = np.linspace(-10, 10, 5)
    #     p3 = np.linspace(0, 1, 10)


    #     # can use meshgrid to create a 3D array of test data
    #     P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
    #     f = np.sqrt(P1) + P2 * P3

    #     # Create regular grid interpolator instance
    #     interp = om.MetaModelStructuredComp(method='scipy_quintic')
    #     interp.add_input('p1', 0.5, training_data=p1)
    #     interp.add_input('p2', 0.0, training_data=p2)
    #     interp.add_input('p3', 3.14, training_data=p3)
    #     # interp.add_input('p4', 4.0, training_data=p4)
    #     # interp.add_input('p5', 10.0, training_data=p5)

    #     interp.add_output('f', 0.0, training_data=f)

    #     # Set up the OpenMDAO model
    #     model = om.Group()
    #     model.add_subsystem('comp', interp, promotes=["*"])
    #     prob = om.Problem(model)
    #     prob.setup()

    #     MetaModelVisualization(interp)

if __name__ == '__main__':
    unittest.main()
