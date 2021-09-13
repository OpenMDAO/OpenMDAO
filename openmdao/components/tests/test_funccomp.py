import itertools
import unittest
import math
import os
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_almost_equal
import scipy
from io import StringIO

from distutils.version import LooseVersion

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_warning
from openmdao.utils.om_warnings import OMDeprecationWarning

_ufunc_test_data = {
    'min': {
        'str': 'f=min(x)',
        'check_func': np.min,
        'args': { 'f': {'val': np.zeros(6)},
                  'x': {'val': np.random.random(6)}}},
    'max': {
        'str': 'f=max(x)',
        'check_func': np.max,
        'args': { 'f': {'val': np.zeros(6)},
                  'x': {'val': np.random.random(6)}}},
    'diff': {
        'str': 'f=diff(x)',
        'check_func': np.diff,
        'args': { 'f': {'val': np.zeros(5)},
                  'x': {'val': np.random.random(6)}}},
    'abs': {
        'str': 'f=abs(x)',
        'check_func': np.abs,
        'args': { 'f': {'val': np.zeros(6)},
                  'x': {'val': np.random.random(6)}}},
    'acos': {
        'str': 'f=acos(x)',
        'check_func': np.arccos,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6) - 0.5}}},
    'arccos': {
        'str': 'f=arccos(x)',
        'check_func': np.arccos,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6) - 0.5}}},
    'arccosh': {
        'str': 'f=arccosh(x)',
        'check_func': np.arccosh,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': 1.1 + np.random.random(6)}}},
    'acosh': {
        'str': 'f=acosh(x)',
        'check_func': np.arccosh,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': 1.1 + np.random.random(6)}}},
    'arange': {
        'str': 'f=arange(0,10,2)',
        'check_val': np.arange(0, 10, 2),
        'args': {'f': {'val': np.zeros(5)}}},
    'arcsin': {
        'str': 'f=arcsin(x)',
        'check_func': np.arcsin,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6) - .5}}},
    'arcsinh': {
        'str': 'f=arcsinh(x)',
        'check_func': np.arcsinh,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'asinh': {
        'str': 'f=asinh(x)',
        'check_func': np.arcsinh,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'asin': {
        'str': 'f=asin(x)',
        'check_func': np.arcsin,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6) - .5}}},
    'arctan': {
        'str': 'f=arctan(x)',
        'check_func': np.arctan,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'arctan2': {
        'str': 'f=arctan2(y, x)',
        'check_val': np.array([-2.35619449, -0.78539816,  0.78539816,  2.35619449]),
        'args': {'f': {'val': np.zeros(4)},
                 'x': {'val': np.array([-1, +1, +1, -1])},
                 'y': {'val': np.array([-1, -1, +1, +1])}}},
    'atan': {
        'str': 'f=atan(x)',
        'check_func': np.arctan,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'cos': {
        'str': 'f=cos(x)',
        'check_func': np.cos,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'cosh': {
        'str': 'f=cosh(x)',
        'check_func': np.cosh,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'dot': {
        'str': 'f=dot(x, y)',
        'check_func': np.dot,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'e': {
        'str': 'f=e',
        'check_val': np.e,
        'args': {'f': {'val': 0.0}}},
    'erf': {
        'str': 'f=erf(x)',
        'check_func': scipy.special.erf,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'erfc': {
        'str': 'f=erfc(x)',
        'check_func': scipy.special.erfc,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'exp': {
        'str': 'f=exp(x)',
        'check_func': np.exp,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'expm1': {
        'str': 'f=expm1(x)',
        'check_func': np.expm1,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'fmax': {
        'str': 'f=fmax(x, y)',
        'check_func': np.fmax,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'fmin': {
        'str': 'f=fmin(x, y)',
        'check_func': np.fmin,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'inner': {
        'str': 'f=inner(x, y)',
        'check_func': np.inner,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'isinf': {
        'str': 'f=isinf(x)',
        'check_func': np.isinf,
        'args': {'f': {'val': np.zeros(3)},
                 'x': {'val': [0, np.inf, 5.0]}}},
    'isnan': {
        'str': 'f=isnan(x)',
        'check_func': np.isnan,
        'args': {'f': {'val': np.zeros(3)},
                 'x': {'val': [0, np.nan, np.nan]}}},
    'kron': {
        'str': 'f=kron(x, y)',
        'check_func': np.kron,
        'args': {'f': {'val': np.zeros(36)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'linspace': {
        'str': 'f=linspace(0,10,50)',
        'check_val': np.linspace(0, 10, 50),
        'args': {'f': {'val': np.zeros(50)}}},
    'log': {
        'str': 'f=log(x)',
        'check_func': np.log,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6) + 0.1}}},
    'log10': {
        'str': 'f=log10(x)',
        'check_func': np.log10,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6) + 0.1}}},
    'log1p': {
        'str': 'f=log1p(x)',
        'check_func': np.log1p,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'matmul': {
        'str': 'f=matmul(x, y)',
        'check_func': np.matmul,
        'args': {'f': {'val': np.zeros((3, 1))},
                 'x': {'val': np.random.random((3, 3))},
                 'y': {'val': np.random.random((3, 1))}}},
    'maximum': {
        'str': 'f=maximum(x, y)',
        'check_func': np.maximum,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'minimum': {
        'str': 'f=minimum(x, y)',
        'check_func': np.minimum,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'ones': {
        'str': 'f=ones(21)',
        'check_val': np.ones(21),
         'args': {'f': {'val': np.zeros(21)}}},
    'outer': {
        'str': 'f=outer(x, y)',
        'check_func': np.outer,
        'args': {'f': {'val': np.zeros((6, 6))},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6)}}},
    'pi': {
        'str': 'f=pi',
        'check_val': np.pi,
        'args': {'f': {'val': 0.0}}},
    'power': {
        'str': 'f=power(x, y)',
        'check_func': np.power,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)},
                 'y': {'val': np.random.random(6) + 1.0}}},
    'prod': {
        'str': 'f=prod(x)',
        'check_func': np.prod,
        'args': {'f': {'val': 0.0},
                 'x': {'val': np.random.random(6)}}},
    'sin': {
        'str': 'f=sin(x)',
        'check_func': np.sin,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'sinh': {
        'str': 'f=sinh(x)',
        'check_func': np.sinh,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'sum': {
        'str': 'f=sum(x)',
        'check_func': np.sum,
        'args': {'f': {'val': 0.0},
                 'x': {'val': np.random.random(6)}}},
    'tan': {
        'str': 'f=tan(x)',
        'check_func': np.tan,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'tanh': {
        'str': 'f=tanh(x)',
        'check_func': np.tanh,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}}},
    'tensordot': {
        'str': 'f=tensordot(x, y)',
        'check_func': np.tensordot,
        'args': {'f': {'val': 0.0},
                 'x': {'val': np.random.random((6, 6))},
                 'y': {'val': np.random.random((6, 6))}}},
    'zeros': {
        'str': 'f=zeros(21)',
        'check_val': np.zeros(21),
        'args': {'f': {'val': np.zeros(21)}}},
}


# 'factorial' will raise a RuntimeError or a deprecation warning depending on scipy version
if LooseVersion(scipy.__version__) >= LooseVersion("1.5.0"):
    _ufunc_test_data['factorial'] = {
        'str': 'f=factorial(x)',
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}},
        'error': (RuntimeError,
                  "The 'factorial' function is not supported for SciPy "
                  f"versions >= 1.5, current version: {scipy.__version__}")
    }
else:
    _ufunc_test_data['factorial'] = {
        'str': 'f=factorial(x)',
        'check_func': scipy.special.factorial,
        'args': {'f': {'val': np.zeros(6)},
                 'x': {'val': np.random.random(6)}},
        'warning': (OMDeprecationWarning,
                    "The 'factorial' function is deprecated. "
                    "It is no longer supported for SciPy versions >= 1.5.")
    }


class TestFuncComp(unittest.TestCase):

    def test_mixed_type(self):
        prob = om.Problem()
        def func(x=np.arange(10, dtype=float)):
            y = np.sum(x)
            return y
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))
        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 45.0, 0.00001)

    def test_simple(self):
        prob = om.Problem()
        def func(x=2.0):
            y = x + 1.
            return y

        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))

        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 3.0, 0.00001)

    # def test_units(self):
    #     prob = om.Problem()
    #     prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
    #     C1 = prob.model.add_subsystem('C1', om.ExecComp('y=x+z+1.',
    #                                                     x={'val': 2.0, 'units': 'm'},
    #                                                     y={'units': 'm'},
    #                                                     z=2.0))
    #     prob.model.connect('indep.x', 'C1.x')

    #     prob.setup()

    #     prob.set_solver_print(level=0)
    #     prob.run_model()

    #     assert_near_equal(C1._outputs['y'], 4.0, 0.00001)

    # def test_units_varname(self):
    #     prob = om.Problem()

    #     with self.assertRaises(TypeError) as cm:
    #         prob.model.add_subsystem('C1', om.ExecComp('y=x+units+1.',
    #                                                    x={'val': 2.0, 'units': 'm'},
    #                                                    y={'units': 'm'},
    #                                                    units=2.0))

    #     self.assertEqual(str(cm.exception),
    #                      "ExecComp: Value (2.0) of option 'units' has type 'float', "
    #                      "but type 'str' was expected.")

    # def test_units_varname_str(self):
    #     prob = om.Problem()

    #     with self.assertRaises(ValueError) as cm:
    #         prob.model.add_subsystem('C1', om.ExecComp('y=x+units+1.',
    #                                                    x={'val': 2.0, 'units': 'm'},
    #                                                    y={'units': 'm'},
    #                                                    units='two'))

    #     self.assertEqual(str(cm.exception), "The units 'two' are invalid.")

    # def test_units_varname_novalue(self):
    #     prob = om.Problem()
    #     prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
    #     C1 = prob.model.add_subsystem('C1', om.ExecComp('y=x+units+1.',
    #                                                     x={'val': 2.0, 'units': 'm'},
    #                                                     y={'units': 'm'}))
    #     prob.model.connect('indep.x', 'C1.x')

    #     with self.assertRaises(NameError) as cm:
    #         prob.setup()

    #     self.assertEqual(str(cm.exception),
    #                      "'C1' <class ExecComp>: cannot use variable name 'units' because it's a reserved keyword.")

    # def test_common_units(self):
    #     # all variables in the ExecComp have the same units
    #     prob = om.Problem()

    #     prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
    #     prob.model.add_subsystem('comp', om.ExecComp('y=x+z+1.', units='m',
    #                                                  x={'val': 2.0},
    #                                                  z=2.0))
    #     prob.model.connect('indep.x', 'comp.x')

    #     prob.setup()
    #     prob.run_model()

    #     assert_near_equal(prob['comp.y'], 4.0, 0.00001)

    # def test_common_units_no_meta(self):
    #     # make sure common units are assigned when no metadata is provided
    #     prob = om.Problem()

    #     prob.model.add_subsystem('indep', om.IndepVarComp('x', 2.0, units='km'))
    #     prob.model.add_subsystem('comp', om.ExecComp('y = x+1', units='m'))

    #     prob.model.connect('indep.x', 'comp.x')

    #     prob.setup()
    #     prob.run_model()

    #     assert_near_equal(prob['comp.y'], 2001., 0.00001)

    # def test_conflicting_units(self):
    #     prob = om.Problem()
    #     prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
    #     C1 = prob.model.add_subsystem('C1', om.ExecComp('y=x+z+1.', units='m',
    #                                                     x={'val': 2.0, 'units': 'km'},
    #                                                     z=2.0))
    #     prob.model.connect('indep.x', 'C1.x')

    #     with self.assertRaises(RuntimeError) as cm:
    #         prob.setup()

    #     self.assertEqual(str(cm.exception),
    #                      "'C1' <class ExecComp>: units of 'km' have been specified for variable 'x', but "
    #                      "units of 'm' have been specified for the entire component.")

    def test_shape_and_value(self):
        p = om.Problem()
        model = p.model

        def func(x:{'shape': (5,)}):
            y = 3.0*x + 2.5
            return y
        model.add_subsystem('comp', om.ExplicitFuncComp(func))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_conflicting_shape(self):
        p = om.Problem()
        model = p.model

        model.add_subsystem('comp', om.ExecComp('y=3.0*x + 2.5',
                                                x={'shape': (5,), 'val': 5},
                                                y={'shape': (5,)}))

        with self.assertRaises(Exception) as context:
            p.setup()

        self.assertEqual(str(context.exception).replace('L,', ','),  # L on Windows
                         "'comp' <class ExecComp>: shape of (5,) has been specified for variable 'x', "
                         "but a value of shape (1,) has been provided.")

    def test_common_shape(self):
        p = om.Problem()
        model = p.model

        model.add_subsystem('comp', om.ExecComp('y=3.0*x + 2.5', shape=(5,)))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_common_shape_with_values(self):
        p = om.Problem()
        model = p.model

        model.add_subsystem('comp', om.ExecComp('y=3.0*x + 2.5', shape=(5,),
                                                x={'val': np.zeros(5)},
                                                y={'val': np.zeros(5)}))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_common_shape_conflicting_shape(self):
        p = om.Problem()
        model = p.model

        model.add_subsystem('comp', om.ExecComp('y=3.0*x + 2.5', shape=(5,),
                                                y={'shape': (10,)}))

        with self.assertRaises(Exception) as context:
            p.setup()

        self.assertEqual(str(context.exception).replace('L,', ','),  # L on Windows
                         "'comp' <class ExecComp>: shape of (10,) has been specified for variable 'y', "
                         "but shape of (5,) has been specified for the entire component.")

    def test_common_shape_conflicting_value(self):
        p = om.Problem()
        model = p.model

        model.add_subsystem('comp', om.ExecComp('y=3.0*x + 2.5', shape=(5,),
                                                x={'val': 5}))

        with self.assertRaises(Exception) as context:
            p.setup()

        self.assertEqual(str(context.exception).replace('1L,', '1,'),  # 1L on Windows
                         "'comp' <class ExecComp>: value of shape (1,) has been specified for variable 'x', "
                         "but shape of (5,) has been specified for the entire component.")

    def test_math(self):
        prob = om.Problem()
        def func(x=2.0):
            y = np.sin(x)
            return y
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))

        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], math.sin(2.0), 0.00001)

    def test_array(self):
        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp('y=x[1]',
                                                        x=np.array([1., 2., 3.]),
                                                        y=0.0))

        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 2.0, 0.00001)

    def test_array_lhs(self):
        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp(['y[0]=x[1]', 'y[1]=x[0]'],
                                                        x=np.array([1., 2., 3.]),
                                                        y=np.array([0., 0.])))

        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], np.array([2., 1.]), 0.00001)

    def test_simple_array_model(self):
        prob = om.Problem()
        def func(x:{'shape': 2}):
            y = np.array([2.0*x[0]+7.0*x[1], 5.0*x[0]-3.0*x[1]])
            return y

        prob.model.add_subsystem('comp', om.ExplicitFuncComp(func))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-5, rtol=1e-5)

    def test_simple_array_model2(self):
        prob = om.Problem()
        def func(x:{'shape': 2}):
            y = np.array([[2., 7.], [5., -3.]]).dot(x)
            return y
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(func))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-5, rtol=1e-5)

    def test_complex_step(self):
        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp(['y=2.0*x+1.'], x=2.0))

        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 5.0, 0.00001)

        C1._linearize()

        assert_near_equal(C1._jacobian[('y', 'x')], [[2.0]], 0.00001)

    def test_complex_step2(self):
        prob = om.Problem(om.Group())
        prob.model.add_subsystem('comp', om.ExecComp('y=x*x + x*2.0', x=2.0))
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        J = prob.compute_totals(['comp.y'], ['comp.x'], return_format='flat_dict')
        assert_near_equal(J['comp.y', 'comp.x'], np.array([[6.0]]), 0.00001)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(['comp.y'], ['comp.x'], return_format='flat_dict')
        assert_near_equal(J['comp.y', 'comp.x'], np.array([[6.0]]), 0.00001)

    def test_abs_complex_step(self):
        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp('y=2.0*abs(x)', x=-2.0))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 4.0, 0.00001)

        # any positive C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = 1.0e-10
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], [[2.0]], 0.00001)

        C1._inputs['x'] = -3.0
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], [[-2.0]], 0.00001)

        C1._inputs['x'] = 0.0
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], [[2.0]], 0.00001)

    def test_arctan_complex_step(self):
        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp('y=2.0*arctan2(y, x)', x=np.array([1+2j]), y=1))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], np.array([1.57079633]), 1e-8)

    def test_abs_array_complex_step(self):
        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp('y=2.0*abs(x)',
                                                        x=np.ones(3)*-2.0, y=np.zeros(3)))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], np.ones(3)*4.0, 0.00001)

        # any positive C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = np.ones(3)*1.0e-10
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], np.eye(3)*2.0, 0.00001)

        C1._inputs['x'] = np.ones(3)*-3.0
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], np.eye(3)*-2.0, 0.00001)

        C1._inputs['x'] = np.zeros(3)
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], np.eye(3)*2.0, 0.00001)

        C1._inputs['x'] = np.array([1.5, -0.6, 2.4])
        C1._linearize()
        expect = np.zeros((3, 3))
        expect[0, 0] = 2.0
        expect[1, 1] = -2.0
        expect[2, 2] = 2.0

        assert_near_equal(C1._jacobian['y', 'x'], expect, 0.00001)

    def test_has_diag_partials_error(self):
        p = om.Problem()
        model = p.model

        def func(x:{'shape': 5}, A=np.arange(15).reshape((3,5))):
            y=A.dot(x)
            return y

        model.add_subsystem('comp', om.ExplicitFuncComp(func, has_diag_partials=True))

        p.setup()

        with self.assertRaises(Exception) as context:
            p.final_setup()

        self.assertEqual(str(context.exception),
                         "'comp' <class ExplicitFuncComp>: has_diag_partials is True but partial(y, x) is not square (shape=(3, 5)).")

    def test_has_diag_partials(self):
        # Really check to see that the has_diag_partials argument had its intended effect

        # run with has_diag_partials=False
        p = om.Problem()
        model = p.model

        def func(x:{'shape': 5}):
            y=3.0*x + 2.5
            return y

        comp = model.add_subsystem('comp', om.ExplicitFuncComp(func, has_diag_partials=False))
        p.setup()

        declared_partials = comp._declared_partials[('y','x')]
        self.assertTrue('rows' not in declared_partials )
        self.assertTrue('cols' not in declared_partials )

        # run with has_diag_partials=True
        p = om.Problem()
        model = p.model
        comp = model.add_subsystem('comp', om.ExplicitFuncComp(func, has_diag_partials=True))
        p.setup()
        p.final_setup()

        declared_partials = comp._declared_partials[('y','x')]
        self.assertTrue('rows' in declared_partials )
        self.assertListEqual([0,1,2,3,4], list( comp._declared_partials[('y','x')]['rows']))
        self.assertTrue('cols' in declared_partials )
        self.assertListEqual([0,1,2,3,4], list( comp._declared_partials[('y','x')]['cols']))

    def test_exec_comp_deriv_sparsity(self):
        # Check to make sure that when an ExecComp has more than one
        # expression that only the partials that are needed are declared and computed

        # with has_diag_partials set to the default of False and just scalars
        p = om.Problem()
        model = p.model
        comp = om.ExecComp(['y1=2.0*x1+1.', 'y2=3.0*x2-1.'],x1=1.0, x2=2.0)
        model.add_subsystem('comp', comp)
        p.setup()
        p.final_setup()

        # make sure only the partials that are needed are declared
        declared_partials = comp._declared_partials
        self.assertListEqual( sorted([('y1', 'x1'), ('y2', 'x2') ]),
                              sorted(declared_partials.keys()))

        p.run_model()

        # make sure only what is needed was computed
        subjacs_info = comp._jacobian._subjacs_info
        self.assertListEqual(sorted([('comp.y1', 'comp.x1'), ('comp.y2', 'comp.x2'),
                                     ('comp.y1', 'comp.y1'),('comp.y2', 'comp.y2')]),
                             sorted(subjacs_info.keys()))

        # make sure the result of compute_partials is correct
        J = p.compute_totals(of=['comp.y1'], wrt=['comp.x1'], return_format='array')
        self.assertEqual(2.0, J)
        J = p.compute_totals(of=['comp.y2'], wrt=['comp.x2'], return_format='array')
        self.assertEqual(3.0, J)

        # make sure this works with arrays and when has_diag_partials is the default of False
        p = om.Problem()
        model = p.model
        comp = om.ExecComp(['y1=2.0*x1+1.', 'y2=3.0*x2-1.'],
                           x1=np.ones(5), y1=np.ones(5), x2=np.ones(5), y2=np.ones(5))
        model.add_subsystem('comp', comp)
        p.setup()
        p.final_setup()

        declared_partials = comp._declared_partials
        self.assertListEqual( sorted([('y1', 'x1'), ('y2', 'x2') ]),
                              sorted(declared_partials.keys()))

        p.run_model()
        J = p.compute_totals(of=['comp.y1'], wrt=['comp.x1'], return_format='array')
        self.assertTrue(np.all(2.0*np.identity(5) == J))
        J = p.compute_totals(of=['comp.y2'], wrt=['comp.x2'], return_format='array')
        self.assertTrue(np.all(3.0*np.identity(5) == J))

        # with has_diag_partials True to make sure that still works with arrays
        p = om.Problem()
        model = p.model
        comp = om.ExecComp(['y1=2.0*x1+1.', 'y2=3.0*x2-1.'], has_diag_partials=True,
                           x1=np.ones(5), y1=np.ones(5), x2=np.ones(5), y2=np.ones(5) )
        model.add_subsystem('comp', comp)
        p.setup()
        p.final_setup()

        declared_partials = comp._declared_partials
        self.assertListEqual( sorted([('y1', 'x1'), ('y2', 'x2') ]),
                              sorted(declared_partials.keys()))
        self.assertTrue('cols' in declared_partials[('y1', 'x1')] )
        self.assertTrue('rows' in declared_partials[('y1', 'x1')] )
        self.assertTrue('cols' in declared_partials[('y2', 'x2')] )
        self.assertTrue('rows' in declared_partials[('y2', 'x2')] )
        self.assertListEqual([0,1,2,3,4], list( comp._declared_partials[('y1','x1')]['rows']))
        self.assertListEqual([0,1,2,3,4], list( comp._declared_partials[('y1','x1')]['cols']))
        self.assertListEqual([0,1,2,3,4], list( comp._declared_partials[('y2','x2')]['rows']))
        self.assertListEqual([0,1,2,3,4], list( comp._declared_partials[('y2','x2')]['cols']))

        p.run_model()

        J = p.compute_totals(of=['comp.y1'], wrt=['comp.x1'], return_format='array')
        self.assertTrue(np.all(2.0*np.identity(5) == J))
        J = p.compute_totals(of=['comp.y2'], wrt=['comp.x2'], return_format='array')
        self.assertTrue(np.all(3.0*np.identity(5) == J))

    def test_has_diag_partials_shape_only(self):
        p = om.Problem()
        model = p.model

        def func(x:{'shape': (5,)}):
            y=3.0*x + 2.5
            return y

        model.add_subsystem('comp', om.ExplicitFuncComp(func, has_diag_partials=True))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_feature_has_diag_partials(self):

        p = om.Problem()
        model = p.model

        model.add_subsystem('comp', om.ExecComp('y=3.0*x + 2.5',
                                                has_diag_partials=True,
                                                x=np.ones(5), y=np.ones(5)))

        p.setup()

        p.set_val('comp.x', np.ones(5))

        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_feature_multi_output(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp(['y1=x+1.', 'y2=x-1.']), promotes=['x'])

        prob.setup()

        prob.set_val('x', 2.0)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.y1'), 3.0, 0.00001)
        assert_near_equal(prob.get_val('comp.y2'), 1.0, 0.00001)

    def test_feature_multi_output2(self):
        # verify that expressions can have multiple LHS variables.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp(['y1, y2 = x+1., x-1.']), promotes=['x'])

        prob.setup()

        prob.set_val('x', 2.0)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.y1'), 3.0, 0.00001)
        assert_near_equal(prob.get_val('comp.y2'), 1.0, 0.00001)

    def test_feature_array(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x[1]',
                                                x=np.array([1., 2., 3.]),
                                                y=0.0))

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.y'), 2.0, 0.00001)

    def test_feature_math(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('z = sin(x)**2 + cos(y)**2'))

        prob.setup()

        prob.set_val('comp.x', np.pi/2.0)
        prob.set_val('comp.y', np.pi/2.0)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.z'), 1.0, 0.00001)

    def test_feature_numpy(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=sum(x)', x=np.array([1., 2., 3.])))

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['comp.y'], 6.0, 0.00001)

    def test_feature_metadata(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('z=x+y',
                                                x={'val': 0.0, 'units': 'inch'},
                                                y={'val': 0.0, 'units': 'inch'},
                                                z={'val': 0.0, 'units': 'inch'}))

        prob.setup()

        prob.set_val('comp.x', 12.0, units='inch')
        prob.set_val('comp.y', 1.0, units='ft')

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.z'), 24.0, 0.00001)

    def test_feature_options(self):

        model = om.Group()

        xcomp = model.add_subsystem('comp', om.ExecComp('y=2*x', shape=(2,)))

        xcomp.options['units'] = 'm'

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('comp.x', [100., 200.], units='cm')

        prob.run_model()

        assert_near_equal(prob.get_val('comp.y'), [2., 4.], 0.00001)

    def test_list_outputs_resids_tol(self):
        prob = om.Problem()
        model = prob.model

        def func(a=2.0, b=5.0, c=3.0, x=np.ones(2)):
            y = a * x ** 2 + b * x + c
            return y

        model.add_subsystem("quad_1", om.ExplicitFuncComp(func))

        balance = model.add_subsystem("balance", om.BalanceComp())
        balance.add_balance("x_1", val=np.array([1, -1]), rhs_val=np.array([0., 0.]))
        model.connect("balance.x_1", "quad_1.x")
        model.connect("quad_1.y", "balance.lhs:x_1")

        prob.model.linear_solver = om.ScipyKrylov()
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=2)

        prob.setup()
        prob.model.nonlinear_solver.options["maxiter"] = 0
        prob.run_model()

        stream = StringIO()
        outputs = prob.model.list_outputs(residuals=True, residuals_tol=1e-5, out_stream=stream)

        text = stream.getvalue()
        self.assertTrue("balance" in text)
        self.assertTrue("x_1" in text)


if __name__ == "__main__":
    unittest.main()
