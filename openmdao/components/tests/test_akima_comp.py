"""
Unit tests for the akima interpolator component.
"""
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials


class AkimaTestCase(unittest.TestCase):

    def test_basic(self):
        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        ncp = len(xcp)
        n = 50
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()

        comp = om.AkimaSplineComp(num_control_points=ncp, num_points=n,
                                  name='chord', input_x=True, input_xcp=True)

        prob.model.add_subsystem('akima', comp)

        prob.setup(force_alloc_complex=True)

        prob['akima.chord:x_cp'] = xcp
        prob['akima.chord:y_cp'] = ycp.reshape((1, ncp))
        prob['akima.chord:x'] = x

        prob.run_model()

        y = np.array([[ 5.        ,  7.20902005,  9.21276849, 10.81097162, 11.80335574,
                        12.1278001 , 12.35869145, 12.58588536, 12.81022332, 13.03254681,
                        13.25369732, 13.47451633, 13.69584534, 13.91852582, 14.14281484,
                        14.36710105, 14.59128625, 14.81544619, 15.03965664, 15.26399335,
                        15.48853209, 15.7133486 , 15.93851866, 16.16573502, 16.39927111,
                        16.63928669, 16.8857123 , 17.1384785 , 17.39751585, 17.66275489,
                        17.93412619, 18.21156029, 18.49498776, 18.78433915, 19.07954501,
                        19.38053589, 19.68724235, 19.99959495, 20.31752423, 20.64096076,
                        20.96983509, 21.37579297, 21.94811407, 22.66809748, 23.51629844,
                        24.47327219, 25.51957398, 26.63575905, 27.80238264, 29.        ]])


        assert_array_almost_equal(y, prob['akima.chord:y'])

        derivs = prob.check_partials(compact_print=True, method='cs')

        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

    def test_fixed_grid(self):
        # Unsupported
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        ncp = len(ycp)
        n = 11

        prob = om.Problem()

        comp = om.AkimaSplineComp(num_control_points=ncp, num_points=n,
                                 name='chord', eval_at='end')

        prob.model.add_subsystem('akima', comp)

        prob.setup(force_alloc_complex=True)

        prob['akima.chord:y_cp'] = ycp.reshape((1, ncp))

        prob.run_model()

        y = np.array([[ 5.        ,  9.4362525 , 12.        , 13.0012475 , 14.        ,
                        14.99875415, 16.        , 17.93874585, 21.        , 24.625    ,
                        29.        ]])

        assert_array_almost_equal(y, prob['akima.chord:y'])

        derivs = prob.check_partials(compact_print=True, method='cs')

        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

        self.assertTrue('akima.chord:x' in prob.model.akima._outputs._names)
        self.assertTrue('akima.chord:x_cp' in prob.model.akima._outputs._names)

    def test_vectorized(self):
        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([[5.0, 12.0, 14.0, 16.0, 21.0, 29.0],
                        [7.0, 13.0, 9.0, 6.0, 12.0, 14.0]])
        ncp = len(xcp)
        n = 12
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()

        comp = om.AkimaSplineComp(num_control_points=ncp, num_points=n, vec_size=2,
                                 name='chord', input_x=True, input_xcp=True)

        prob.model.add_subsystem('akima', comp)

        prob.setup(force_alloc_complex=True)

        prob['akima.chord:x_cp'] = xcp
        prob['akima.chord:y_cp'] = ycp
        prob['akima.chord:x'] = x

        prob.run_model()

        y = np.array([[ 5.        , 12.        , 13.01239669, 14.        , 14.99888393,
                        16.        , 17.06891741, 18.26264881, 19.5750558 , 21.        ,
                        24.026042, 29.        ],
                      [ 7.        , 13.        , 11.02673797,  9.        ,  7.09090909,
                        6.        ,  6.73660714,  8.46428571, 10.45982143, 12.        ,
                        13.08035714, 14.        ]])

        assert_array_almost_equal(y, prob['akima.chord:y'])

        derivs = prob.check_partials(compact_print=True, method='cs')

        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)


class TestAkimaFeature(unittest.TestCase):

    def test_input_grid(self):
        # Same as test_basic
        import numpy as np

        import openmdao.api as om

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        ncp = len(xcp)
        n = 50
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()

        comp = om.AkimaSplineComp(num_control_points=ncp, num_points=n,
                                  name='chord', input_x=True, input_xcp=True)

        prob.model.add_subsystem('akima', comp)

        prob.setup(force_alloc_complex=True)

        prob['akima.chord:x_cp'] = xcp
        prob['akima.chord:y_cp'] = ycp.reshape((1, ncp))
        prob['akima.chord:x'] = x

        prob.run_model()

        y = np.array([[ 5.        ,  7.20902005,  9.21276849, 10.81097162, 11.80335574,
                        12.1278001 , 12.35869145, 12.58588536, 12.81022332, 13.03254681,
                        13.25369732, 13.47451633, 13.69584534, 13.91852582, 14.14281484,
                        14.36710105, 14.59128625, 14.81544619, 15.03965664, 15.26399335,
                        15.48853209, 15.7133486 , 15.93851866, 16.16573502, 16.39927111,
                        16.63928669, 16.8857123 , 17.1384785 , 17.39751585, 17.66275489,
                        17.93412619, 18.21156029, 18.49498776, 18.78433915, 19.07954501,
                        19.38053589, 19.68724235, 19.99959495, 20.31752423, 20.64096076,
                        20.96983509, 21.37579297, 21.94811407, 22.66809748, 23.51629844,
                        24.47327219, 25.51957398, 26.63575905, 27.80238264, 29.        ]])


        assert_rel_error(self, prob['akima.chord:y'], y, 1e-6)

    def test_fixed_grid(self):
        # Unsupported
        import numpy as np

        import openmdao.api as om

        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        ncp = len(ycp)
        n = 11

        prob = om.Problem()

        comp = om.AkimaSplineComp(num_control_points=ncp, num_points=n,
                                  name='chord', eval_at='end')

        prob.model.add_subsystem('akima', comp)

        prob.setup(force_alloc_complex=True)

        prob['akima.chord:y_cp'] = ycp.reshape((1, ncp))

        prob.run_model()

        y = np.array([[ 5.        ,  9.4362525 , 12.        , 13.0012475 , 14.        ,
                        14.99875415, 16.        , 17.93874585, 21.        , 24.625    ,
                        29.        ]])

        assert_rel_error(self, prob['akima.chord:y'], y, 1e-6)


if __name__ == '__main__':
    unittest.main()
