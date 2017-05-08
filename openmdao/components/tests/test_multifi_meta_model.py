import numpy as np
import unittest

from openmdao.api import Group, Problem, MetaModel, MultiFiMetaModel, IndepVarComp, \
     ResponseSurface, FloatKrigingSurrogate, KrigingSurrogate, MultiFiSurrogateModel
from openmdao.core.component import _NotSet

#from openmdao.api import

from openmdao.test.util import assert_rel_error

from six.moves import cStringIO
from re import findall

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
        mm = MultiFiMetaModel(nfi=3)

        mm.add_param('x', 0.)
        mm.add_output('y', 0.)

        prob = Problem(Group())
        prob.root.add('mm', mm)
        prob.setup(check=False)

        self.assertEqual(prob['mm.train:x'], [])
        self.assertEqual(prob['mm.train:x_fi2'], [])
        self.assertEqual(prob['mm.train:x_fi3'], [])
        self.assertEqual(prob['mm.train:y'], [])
        self.assertEqual(prob['mm.train:y_fi2'], [])
        self.assertEqual(prob['mm.train:y_fi3'], [])

    def test_one_dim_one_fidelity_training(self):

        mm = MultiFiMetaModel()

        mm.add_param('x', 0.)
        surr = MockSurrogate()
        mm.add_output('y', 0., surrogate = surr)

        prob = Problem(Group())
        prob.root.add('mm', mm)
        prob.setup(check=False)

        prob['mm.train:x'] = [0.0, 0.4, 1.0]
        prob['mm.train:y'] = [3.02720998, 0.11477697, 15.82973195]

        expected_xtrain=[np.array([ [0.0], [0.4], [1.0] ])]
        expected_ytrain=[np.array([ [3.02720998], [0.11477697], [15.82973195] ])]

        prob.run()
        np.testing.assert_array_equal(surr.xtrain, expected_xtrain)
        np.testing.assert_array_equal(surr.ytrain, expected_ytrain)

        expected_xpredict=0.5
        prob['mm.x'] = expected_xpredict
        prob.run()

        np.testing.assert_array_equal(surr.xpredict, expected_xpredict)

    def test_one_dim_bi_fidelity_training(self):

        mm = MultiFiMetaModel(nfi=2)
        mm.add_param('x', 0.)
        surr = MockSurrogate()
        mm.add_output('y', 0., surrogate = surr)

        prob = Problem(Group())
        prob.root.add('mm', mm)
        prob.setup(check=False)

        prob['mm.train:x']= [0.0, 0.4, 1.0]
        prob['mm.train:x_fi2'] = [0.1, 0.2, 0.3, 0.5, 0.6,
                                  0.7, 0.8, 0.9, 0.0, 0.4, 1.0]
        prob['mm.train:y'] = [3.02720998, 0.11477697, 15.82973195]
        prob['mm.train:y_fi2'] = [-9.32828839, -8.31986355, -7.00778837,
                                  -4.54535129, -4.0747189 , -5.30287702,
                                  -4.47456522, 1.85597517, -8.48639501,
                                  -5.94261151, 7.91486597]
        expected_xtrain=[np.array([[0.0], [0.4], [1.0]]),
                         np.array([[0.1], [0.2], [0.3], [0.5], [0.6], [0.7],
                                   [0.8], [0.9], [0.0], [0.4], [1.0]])]
        expected_ytrain=[np.array([[  3.02720998], [0.11477697], [15.82973195]]),
                         np.array([[-9.32828839], [-8.31986355], [-7.00778837], [-4.54535129],
                                   [-4.0747189], [-5.30287702], [-4.47456522], [1.85597517],
                                   [-8.48639501], [-5.94261151],  [7.91486597]])]
        prob.run()
        np.testing.assert_array_equal(surr.xtrain[0], expected_xtrain[0])
        np.testing.assert_array_equal(surr.xtrain[1], expected_xtrain[1])
        np.testing.assert_array_equal(surr.ytrain[0], expected_ytrain[0])
        np.testing.assert_array_equal(surr.ytrain[1], expected_ytrain[1])

    def test_two_dim_bi_fidelity_training(self):
        mm = MultiFiMetaModel(nfi=2)
        mm.add_param('x1', 0.)
        mm.add_param('x2', 0.)
        surr_y1 = MockSurrogate()
        surr_y2 = MockSurrogate()
        mm.add_output('y1', 0., surrogate = surr_y1)
        mm.add_output('y2', 0., surrogate = surr_y2)

        prob = Problem(Group())
        prob.root.add('mm', mm)
        prob.setup(check=False)

        prob['mm.train:x1']     = [1.0, 2.0, 3.0]
        prob['mm.train:x1_fi2'] = [1.1, 2.1, 3.1, 1.0, 2.0, 3.0]
        prob['mm.train:x2']     = [1.0, 2.0, 3.0]
        prob['mm.train:x2_fi2'] = [2.1, 2.2, 2.3, 1.0, 2.0, 3.0]
        prob['mm.train:y1']     = [0.0, 0.1, 0.2]
        prob['mm.train:y1_fi2'] = [3.0, 3.1, 3.3, 3.4, 3.5 ,3.6]
        prob['mm.train:y2']     = [4.0, 4.0, 4.0]
        prob['mm.train:y2_fi2'] = [4.0, 4.1, 4.3, 4.4, 4.5 ,4.6]

        prob.run()
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

    def test_multifidelity_warm_start(self):
        mm = MultiFiMetaModel(nfi=2)
        mm.add_param('x', 0.)
        surr = MockSurrogate()
        mm.add_output('y', 0., surrogate = surr)
        mm.warm_restart=True

        prob = Problem(Group())
        prob.root.add('mm', mm)
        prob.setup(check=False)

        prob['mm.train:x']     = [0.0, 0.4, 1.0]
        prob['mm.train:x_fi2'] = [0.1, 0.2, 0.3, 0.5, 0.6]
        prob['mm.train:y']     = [1.0, 1.4, 2.0]
        prob['mm.train:y_fi2'] = [1.1, 1.2, 1.3, 1.5, 1.6]

        prob.run()
        expected_xtrain=[np.array([[0.0], [0.4], [1.0]]),
                         np.array([[0.1], [0.2], [0.3], [0.5], [0.6]])]
        expected_ytrain=[np.array([[1.0], [1.4], [2.0]]),
                         np.array([[1.1], [1.2], [1.3], [1.5], [1.6]])]
        np.testing.assert_array_equal(surr.xtrain[0], expected_xtrain[0])
        np.testing.assert_array_equal(surr.xtrain[1], expected_xtrain[1])

        np.testing.assert_array_equal(surr.ytrain[0], expected_ytrain[0])
        np.testing.assert_array_equal(surr.ytrain[1], expected_ytrain[1])

        # Test adding only one lowest fidelity sample
        prob['mm.train:x'] = []
        prob['mm.train:y'] = []
        prob['mm.train:x_fi2'] = [2.0]
        prob['mm.train:y_fi2'] = [1.0]
        mm.train=True

        prob.run()
        expected_xtrain=[np.array([[0.0], [0.4], [1.0]]),
                         np.array([[0.1], [0.2], [0.3], [0.5], [0.6], [2.0]])]
        expected_ytrain=[np.array([[1.0], [1.4], [2.0]]),
                         np.array([[1.1], [1.2], [1.3], [1.5], [1.6], [1.0]])]

        np.testing.assert_array_equal(surr.xtrain[0], expected_xtrain[0])
        np.testing.assert_array_equal(surr.xtrain[1], expected_xtrain[1])
        np.testing.assert_array_equal(surr.ytrain[0], expected_ytrain[0])
        np.testing.assert_array_equal(surr.ytrain[1], expected_ytrain[1])

        # Test adding high and low fidelity points
        prob['mm.train:x']     = [3.0]
        prob['mm.train:x_fi2'] = [3.0]
        prob['mm.train:y']     = [4.0]
        prob['mm.train:y_fi2'] = [4.0]

        mm.train = True
        prob.run()

        expected_xtrain=[np.array([[0.0], [0.4], [1.0], [3.0]]),
                         np.array([[0.1], [0.2], [0.3], [0.5], [0.6], [2.0], [3.0]])]
        expected_ytrain=[np.array([[1.0], [1.4], [2.0], [4.0]]),
                         np.array([[1.1], [1.2], [1.3], [1.5], [1.6], [1.0], [4.0]])]
        np.testing.assert_array_equal(surr.xtrain[0], expected_xtrain[0])
        np.testing.assert_array_equal(surr.xtrain[1], expected_xtrain[1])
        np.testing.assert_array_equal(surr.ytrain[0], expected_ytrain[0])
        np.testing.assert_array_equal(surr.ytrain[1], expected_ytrain[1])

if __name__ == "__main__":
    unittest.main()
