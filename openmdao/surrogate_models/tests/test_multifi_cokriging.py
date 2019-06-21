import unittest
from numpy import array, sin, cos, pi, ones
from openmdao.api import MultiFiCoKrigingSurrogate
from openmdao.utils.assert_utils import assert_rel_error

class CoKrigingSurrogateTest(unittest.TestCase):

    def test_1d_1fi_cokriging(self):
        # CoKrigingSurrogate with one fidelity could be used as a KrigingSurrogate
        # Same test as for KrigingSurrogate...  well with predicted test value adjustment

        x = array([[0.05], [.25], [0.61], [0.95]])
        y = array([0.738513784857542, -0.210367746201974, -0.489015457891476, 12.3033138316612])
        krig1 = MultiFiCoKrigingSurrogate()
        krig1.train(x, y)
        new_x = array([0.5])

        mu, sigma = krig1.predict(x[0])
        assert_rel_error(self, mu, [[y[0]]], 1e-4)
        assert_rel_error(self, sigma, [[0.]], 1e-4)

        mu, sigma = krig1.predict(new_x)
        assert_rel_error(self, mu, [[-2.0279]], 1e-3)
        assert_rel_error(self, sigma, [[1.3408]], 1e-3)

        # Test with theta setting instead of estimation
        krig2 = MultiFiCoKrigingSurrogate(theta=0.1, normalize=False)
        krig2.train(x, y)

        mu, sigma = krig2.predict(x[0])
        assert_rel_error(self , mu, [[y[0]]], 1e-4)
        assert_rel_error(self, sigma, [[.0]], 1e-4)

        mu, sigma = krig2.predict(new_x)
        assert_rel_error(self, mu, [[-1.2719]], 1e-3)
        assert_rel_error(self, sigma, [[0.0439]], 1e-3)

    def test_1d_2fi_cokriging(self):
        # Example from Forrester: Engineering design via surrogate modelling
        def f_expensive(x):
            return ((x*6-2)**2)*sin((x*6-2)*2)
        def f_cheap(x):
            return 0.5*((x*6-2)**2)*sin((x*6-2)*2)+(x-0.5)*10. - 5


        x = array([[[0.0], [0.4], [0.6], [1.0]],
                   [[0.1], [0.2], [0.3], [0.5], [0.7],
                    [0.8], [0.9], [0.0], [0.4], [0.6], [1.0]]])
        y = array([[f_expensive(v) for v in array(x[0]).ravel()],
                   [f_cheap(v) for v in array(x[1]).ravel()]])

        cokrig = MultiFiCoKrigingSurrogate()
        cokrig.train_multifi(x, y)

        new_x = array([0.75])
        mu, sigma = cokrig.predict(new_x)
        assert_rel_error(self, mu,  [[f_expensive(new_x[0])]], 0.05)
        assert_rel_error(self, sigma, [[0.]], 0.02)

    def test_2d_1fi_cokriging(self):
        # CoKrigingSurrogate with one fidelity could be used as a KrigingSurrogate
        # Same test as for KrigingSurrogate...  well with predicted test value adjustment

        def branin(x):
            y = (x[1]-(5.1/(4.*pi**2.))*x[0]**2.+5.*x[0]/pi-6.)**2.+10.*(1.-1./(8.*pi))*cos(x[0])+10.
            return y

        x = array([[-2., 0.], [-0.5, 1.5], [1., 3.], [8.5, 4.5],
                   [-3.5, 6.], [4., 7.5], [-5., 9.], [5.5, 10.5],
                   [10., 12.], [7., 13.5], [2.5, 15.]])
        y = array([branin(case) for case in x])
        krig1 = MultiFiCoKrigingSurrogate()
        krig1.train(x, y)

        mu, sigma = krig1.predict([-2., 0.])
        assert_rel_error(self, mu, [[branin(x[0])]], 1e-5)
        assert_rel_error(self, sigma, [[0.]], 1e-5)

        mu, sigma = krig1.predict([5., 5.])
        assert_rel_error(self, mu, [[22]], 1)
        assert_rel_error(self, sigma, [[13]], 1)

        # Test with theta setting instead of estimation
        krig2 = MultiFiCoKrigingSurrogate(theta=[0.1])
        krig1.train(x, y)

        mu, sigma = krig1.predict([-2., 0.])
        assert_rel_error(self, mu, [[branin(x[0])]], 1e-5)
        assert_rel_error(self, sigma, [[0.]], 1e-5)

        mu, sigma = krig1.predict([5., 5.])
        assert_rel_error(self, mu, [[22]], 1)
        assert_rel_error(self, sigma, [[13]], 1)

    def test_2d_2fi_cokriging(self):

        def branin(x):
            x1 = 15*x[0]-5
            x2 = 15*x[1]
            return (x2-(5.1/(4.*pi**2.))*x1**2.+5.*x1/pi-6.)**2.+10.*(1.-1./(8.*pi))*cos(x1)+10.

        # Add a linear error
        def branin_low_fidelity(x):
            return branin(x)+30.*x[1] + 10.

        x = [[[ 0.13073587,  0.24909577],  # expensive (hifi) doe
              [ 0.91915571,  0.4735261 ],
              [ 0.75830543,  0.13321705],
              [ 0.51760477,  0.34594101],
              [ 0.03531219,  0.77765831],
              [ 0.27249206,  0.5306115 ],
              [ 0.62762489,  0.65778471],
              [ 0.3914706 ,  0.09852519],
              [ 0.86565585,  0.85350002],
              [ 0.40806563,  0.91465314]],

             [[ 0.91430235,  0.17029894],  # cheap (lowfi) doe
              [ 0.99329651,  0.76431519],
              [ 0.2012252 ,  0.35006032],
              [ 0.61707854,  0.90210676],
              [ 0.15113004,  0.0133355 ],
              [ 0.07108082,  0.55344447],
              [ 0.4483159 ,  0.52182902],
              [ 0.5926638 ,  0.06595122],
              [ 0.66305449,  0.48579608],
              [ 0.47965045,  0.7407793 ],
              [ 0.13073587,  0.24909577],  # notice hifi doe inclusion
              [ 0.91915571,  0.4735261 ],
              [ 0.75830543,  0.13321705],
              [ 0.51760477,  0.34594101],
              [ 0.03531219,  0.77765831],
              [ 0.27249206,  0.5306115 ],
              [ 0.62762489,  0.65778471],
              [ 0.3914706 ,  0.09852519],
              [ 0.86565585,  0.85350002],
              [ 0.40806563,  0.91465314]]]
        y = array([[branin(case) for case in x[0]],
                   [branin_low_fidelity(case) for case in x[1]]])
        nfi=2
        cokrig = MultiFiCoKrigingSurrogate(normalize=False)
        cokrig.train_multifi(x, y)

        mu, sigma = cokrig.predict([2./3., 1/3.])
        assert_rel_error(self, mu, [[26]], 0.2)
        assert_rel_error(self, sigma, [[0.3]], 0.2)

        # Test with theta setting instead of theta estimation
        cokrig2 = MultiFiCoKrigingSurrogate(theta=0.1, normalize=False)
        cokrig2.train_multifi(x, y)

        mu, sigma = cokrig2.predict([2./3., 1/3.])
        assert_rel_error(self, mu, [[21.7]], 0.1)
        assert_rel_error(self, sigma, [[2.29]], 0.1)

        # Test with theta setting instead of theta estimation
        cokrig2 = MultiFiCoKrigingSurrogate(theta=[0.1, 10], normalize=False)
        cokrig2.train_multifi(x, y)

        mu, sigma = cokrig2.predict([2./3., 1/3.])
        assert_rel_error(self, mu, [[21.01]], 0.2)
        assert_rel_error(self, sigma, [[2.29]], 0.2)

        # Test bad theta setting
        cokrig3 = MultiFiCoKrigingSurrogate(theta=[0.1])
        try:
            cokrig3.train_multifi(x, y)
        except ValueError as err:
            self.assertEqual(str(err),
                "theta must be a list of 2 element(s).")
        else:
            self.fail("ValueError Expected")


if __name__ == "__main__":
    unittest.main()
