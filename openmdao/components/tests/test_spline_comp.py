import openmdao.api as om
import numpy as np
from numpy.testing import assert_array_almost_equal
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials


def test_simple_spline():
    # Data
    x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
    x = np.linspace(1.0, 12.0, 20)

    prob = om.Problem()

    comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp=x)
    prob.model.add_subsystem('akima1', comp)

    comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=y_cp)

    y_interp = prob['akima1.y_val']


def test_multiple_splines():
    x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
    y_cp2 = np.array([1.0, 5.0, 7.0, 8.0, 13.0, 16.0])
    x = np.linspace(1.0, 12.0, 50)

    prob = om.Problem()

    comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp=x, x_interp_name='x_val')
    prob.model.add_subsystem('akima1', comp)

    comp.add_spline(y_cp_name='ycp1', y_interp_name='y_val1', y_cp_val=y_cp)
    comp.add_spline(y_cp_name='ycp2', y_interp_name='y_val2', y_cp_val=y_cp2)

    y_interp = prob['akima1.y_val1']

def test_akima_interp_options():
    x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
    x = np.linspace(1.0, 12.0, 50)

    prob = om.Problem()

    akima_option = {'delta_x': 0.2, 'eps': 1e-30}
    comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp=x, x_cp_name='xcp',
                         x_interp_name='x_val', x_units='km', interp_options=akima_option)

    prob.model.add_subsystem('atmosphere', comp)

    comp.add_spline(y_cp_name='alt_cp', y_interp_name='alt', y_cp_val=y_cp, y_units='kft')

    y_interp = prob['atmosphere.alt']


def test_bspline_interp_options():

    x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    y_cp2 = np.array([1.0, 5.0, 7.0, 8.0, 13.0, 16.0])
    x = np.linspace(1.0, 12.0, 50)

    prob = om.Problem()

    bspline_options = {'order': 5}
    comp = om.SplineComp(method='bspline', x_cp_val=x_cp, x_interp=x, x_cp_name='xcp',
                         x_interp_name='x_val', x_units='km',
                         interp_options=bspline_options)

    prob.model.add_subsystem('atmosphere', comp)

    comp.add_spline(y_cp_name='temp_cp', y_interp_name='temp', y_cp_val=y_cp2, y_units='C')

    y_interp = prob['atmosphere.temp']







def my_test_func():

    x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
    n = 50
    x = np.linspace(1.0, 12.0, n)

    prob = om.Problem()

    comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp=x, vec_size=n, interp_options={'delta_x': 0.10})

    comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=y_cp, y_units='km')
    # comp.add_spline(y_cp_name='ycp2', y_interp_name='y_val2', y_cp_val=y_cp2)

    prob.model.add_subsystem('akima1', comp)

    prob.setup(force_alloc_complex=True)

    prob.run_model()

    print('XCP: \n', prob['akima1.x_cp'])
    print('YCP: \n', prob['akima1.ycp'])
    print('\n')
    print('X_Interp: \n', prob['akima1.x_interp'])
    print('Y_interp: \n', prob['akima1.y_val'])


    # Verification array from AkimaSplineComp
    akima_y = np.array([[ 5.        ,  7.20902005,  9.21276849, 10.81097162, 11.80335574,
                        12.1278001 , 12.35869145, 12.58588536, 12.81022332, 13.03254681,
                        13.25369732, 13.47451633, 13.69584534, 13.91852582, 14.14281484,
                        14.36710105, 14.59128625, 14.81544619, 15.03965664, 15.26399335,
                        15.48853209, 15.7133486 , 15.93851866, 16.16573502, 16.39927111,
                        16.63928669, 16.8857123 , 17.1384785 , 17.39751585, 17.66275489,
                        17.93412619, 18.21156029, 18.49498776, 18.78433915, 19.07954501,
                        19.38053589, 19.68724235, 19.99959495, 20.31752423, 20.64096076,
                        20.96983509, 21.37579297, 21.94811407, 22.66809748, 23.51629844,
                        24.47327219, 25.51957398, 26.63575905, 27.80238264, 29.        ]])

    # assert_array_almost_equal(akima_y.flatten(), prob['akima1.y_val'])

    derivs = prob.check_partials(compact_print=False, method='cs')

    assert_check_partials(derivs, atol=1e-14, rtol=1e-14)



my_test_func()
