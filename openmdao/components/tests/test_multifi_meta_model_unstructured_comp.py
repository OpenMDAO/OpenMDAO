import numpy as np
import unittest

from openmdao.api import Group, Problem, MultiFiMetaModelUnStructuredComp, MultiFiSurrogateModel


class MockSurrogate(MultiFiSurrogateModel):

    def __init__(self):
        super(MockSurrogate, self).__init__()
        self.xtrain = None
        self.ytrain = None

    def train_multifi(self, x, y):
        self.xtrain = x
        self.ytrain = y

    def predict(self, x):
        self.xpredict = x
        return 0.0


class MultiFiMetaModelTestCase(unittest.TestCase):

    def test_inputs_wrt_nfidelity(self):
        mm = MultiFiMetaModelUnStructuredComp(nfi=3)

        mm.add_input('x', 0.)
        mm.add_output('y', 0.)

        prob = Problem(Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        self.assertEqual(mm.metadata['train:x'], None)
        self.assertEqual(mm.metadata['train:x_fi2'], None)
        self.assertEqual(mm.metadata['train:x_fi3'], None)
        self.assertEqual(mm.metadata['train:y'], None)
        self.assertEqual(mm.metadata['train:y_fi2'], None)
        self.assertEqual(mm.metadata['train:y_fi3'], None)

    def test_one_dim_one_fidelity_training(self):
        mm = MultiFiMetaModelUnStructuredComp()
        surr = MockSurrogate()

        mm.add_input('x', 0.)
        mm.add_output('y', 0., surrogate=surr)

        prob = Problem(Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.metadata['train:x'] = [0.0, 0.4, 1.0]
        mm.metadata['train:y'] = [3.02720998, 0.11477697, 15.82973195]

        expected_xtrain=[np.array([[0.0], [0.4], [1.0]])]
        expected_ytrain=[np.array([[3.02720998], [0.11477697], [15.82973195]])]

        prob.run_model()
        np.testing.assert_array_equal(surr.xtrain, expected_xtrain)
        np.testing.assert_array_equal(surr.ytrain, expected_ytrain)

        expected_xpredict=0.5
        prob['mm.x'] = expected_xpredict
        prob.run_model()

        np.testing.assert_array_equal(surr.xpredict, expected_xpredict)

    def test_one_dim_bi_fidelity_training(self):
        mm = MultiFiMetaModelUnStructuredComp(nfi=2)
        surr = MockSurrogate()

        mm.add_input('x', 0.)
        mm.add_output('y', 0., surrogate=surr)

        prob = Problem(Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.metadata['train:x']= [0.0, 0.4, 1.0]
        mm.metadata['train:x_fi2'] = [0.1, 0.2, 0.3, 0.5, 0.6,
                                  0.7, 0.8, 0.9, 0.0, 0.4, 1.0]
        mm.metadata['train:y'] = [3.02720998, 0.11477697, 15.82973195]
        mm.metadata['train:y_fi2'] = [-9.32828839, -8.31986355, -7.00778837,
                                  -4.54535129, -4.0747189 , -5.30287702,
                                  -4.47456522, 1.85597517, -8.48639501,
                                  -5.94261151, 7.91486597]
        expected_xtrain=[np.array([[0.0], [0.4], [1.0]]),
                         np.array([[0.1], [0.2], [0.3], [0.5], [0.6], [0.7],
                                   [0.8], [0.9], [0.0], [0.4], [1.0]])]
        expected_ytrain=[np.array([[ 3.02720998], [0.11477697], [15.82973195]]),
                         np.array([[-9.32828839], [-8.31986355], [-7.00778837], [-4.54535129],
                                   [-4.0747189],  [-5.30287702], [-4.47456522], [1.85597517],
                                   [-8.48639501], [-5.94261151], [7.91486597]])]
        prob.run_model()
        np.testing.assert_array_equal(surr.xtrain[0], expected_xtrain[0])
        np.testing.assert_array_equal(surr.xtrain[1], expected_xtrain[1])
        np.testing.assert_array_equal(surr.ytrain[0], expected_ytrain[0])
        np.testing.assert_array_equal(surr.ytrain[1], expected_ytrain[1])

    def test_two_dim_bi_fidelity_training(self):
        mm = MultiFiMetaModelUnStructuredComp(nfi=2)
        surr_y1 = MockSurrogate()
        surr_y2 = MockSurrogate()

        mm.add_input('x1', 0.)
        mm.add_input('x2', 0.)
        mm.add_output('y1', 0., surrogate=surr_y1)
        mm.add_output('y2', 0., surrogate=surr_y2)

        prob = Problem(Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        mm.metadata['train:x1']     = [1.0, 2.0, 3.0]
        mm.metadata['train:x1_fi2'] = [1.1, 2.1, 3.1, 1.0, 2.0, 3.0]
        mm.metadata['train:x2']     = [1.0, 2.0, 3.0]
        mm.metadata['train:x2_fi2'] = [2.1, 2.2, 2.3, 1.0, 2.0, 3.0]
        mm.metadata['train:y1']     = [0.0, 0.1, 0.2]
        mm.metadata['train:y1_fi2'] = [3.0, 3.1, 3.3, 3.4, 3.5 ,3.6]
        mm.metadata['train:y2']     = [4.0, 4.0, 4.0]
        mm.metadata['train:y2_fi2'] = [4.0, 4.1, 4.3, 4.4, 4.5 ,4.6]

        prob.run_model()
        expected_xtrain=[np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
                         np.array([[1.1, 2.1], [2.1, 2.2], [3.1, 2.3],
                                   [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])]
        expected_y1train=[np.array([[0.0], [0.1], [0.2]]),
                          np.array([[3.0], [3.1], [3.3], [3.4], [3.5], [3.6]])]
        expected_y2train=[np.array([[4.0], [4.0], [4.0]]),
                          np.array([[4.0], [4.1], [4.3], [4.4], [4.5], [4.6]])]

        np.testing.assert_array_equal(surr_y1.ytrain[0], expected_y1train[0])
        np.testing.assert_array_equal(surr_y1.ytrain[1], expected_y1train[1])
        np.testing.assert_array_equal(surr_y2.ytrain[0], expected_y2train[0])
        np.testing.assert_array_equal(surr_y2.ytrain[1], expected_y2train[1])
        np.testing.assert_array_equal(surr_y1.ytrain[0], expected_y1train[0])
        np.testing.assert_array_equal(surr_y1.ytrain[1], expected_y1train[1])
        np.testing.assert_array_equal(surr_y2.ytrain[0], expected_y2train[0])
        np.testing.assert_array_equal(surr_y2.ytrain[1], expected_y2train[1])

    def test_multifi_meta_model_unstructured_deprecated(self):
        # run same test as above, only with the deprecated component,
        # to ensure we get the warning and the correct answer.
        # self-contained, to be removed when class name goes away.
        from openmdao.components.multifi_meta_model_unstructured_comp import MultiFiMetaModelUnStructured  # deprecated
        import warnings

        with warnings.catch_warnings(record=True) as w:
            mm = MultiFiMetaModelUnStructured(nfi=3)

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), "'MultiFiMetaModelUnStructured' has been deprecated. Use "
                                            "'MultiFiMetaModelUnStructuredComp' instead.")

        mm.add_input('x', 0.)
        mm.add_output('y', 0.)

        prob = Problem(Group())
        prob.model.add_subsystem('mm', mm)
        prob.setup(check=False)

        self.assertEqual(mm.metadata['train:x'], None)
        self.assertEqual(mm.metadata['train:x_fi2'], None)
        self.assertEqual(mm.metadata['train:x_fi3'], None)
        self.assertEqual(mm.metadata['train:y'], None)
        self.assertEqual(mm.metadata['train:y_fi2'], None)
        self.assertEqual(mm.metadata['train:y_fi3'], None)

if __name__ == "__main__":
    unittest.main()
