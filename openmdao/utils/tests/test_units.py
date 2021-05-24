""" Unit tests for the units library."""

import os
import unittest
import warnings

import openmdao.api as om
from openmdao.utils.units import NumberDict, PhysicalUnit, _find_unit, import_library, \
    add_unit, add_offset_unit, unit_conversion, get_conversion, simplify_unit
from openmdao.utils.assert_utils import assert_warning, assert_near_equal
from openmdao.warnings import OMDeprecationWarning


class TestNumberDict(unittest.TestCase):

    def test_UnknownKeyGives0(self):
        # a NumberDict instance should initilize using integer and non-integer indices
        # a NumberDict instance should initilize all entries with an initial
        # value of 0
        x = NumberDict()

        # integer test
        self.assertEqual(x[0], 0)

        # string test
        self.assertEqual(x['t'], 0)

    def test__add__KnownValues(self):
        # __add__ should give known result with known input
        # for non-string data types, addition must be commutative

        x = NumberDict()
        y = NumberDict()
        x['t1'], x['t2'] = 1, 2
        y['t1'], y['t2'] = 2, 1

        result1, result2 = x + y, y + x
        self.assertEqual((3, 3), (result1['t1'], result1['t2']))
        self.assertEqual((3, 3), (result2['t1'], result2['t2']))

    def test__sub__KnownValues(self):
        # __sub__ should give known result with known input
        # commuting the input should result in equal magnitude, opposite sign

        x = NumberDict()
        y = NumberDict()
        x['t1'], x['t2'] = 1, 2
        y['t1'], y['t2'] = 2, 1

        result1, result2 = x - y, y - x
        self.assertEqual((-1, 1), (result1['t1'], result1['t2']))
        self.assertEqual((1, -1), (result2['t1'], result2['t2']))

    def test__mul__KnownValues(self):
        # __mul__ should give known result with known input

        x = NumberDict([('t1', 1), ('t2', 2)])
        y = 10

        result1, result2 = x * y, y * x
        self.assertEqual((10, 20), (result1['t1'], result1['t2']))
        self.assertEqual((10, 20), (result2['t1'], result2['t2']))

    def test__div__KnownValues(self):
        # __div__ should give known result with known input

        x = NumberDict()
        x = NumberDict([('t1', 1), ('t2', 2)])
        y = 10.0
        result1 = x / y
        self.assertEqual((.1, .20), (result1['t1'], result1['t2']))


with open(os.path.join(os.path.dirname(__file__),
                       '../unit_library.ini')) as default_lib:
    _unitLib = import_library(default_lib)


def _get_powers(**powdict):
    powers = [0] * len(_unitLib.base_types)
    for name, power in powdict.items():
        powers[_unitLib.base_types[name]] = power
    return powers


class TestPhysicalUnit(unittest.TestCase):

    def test_repr_str(self):
        # __repr__should return a string which could be used to contruct the
        # unit instance, __str__ should return a string with just the unit
        # name for str

        u = _find_unit('d')

        self.assertEqual(repr(u),
                         "PhysicalUnit({'d': 1},86400.0,%s,0.0)" % _get_powers(time=1))
        self.assertEqual(str(u), "<PhysicalUnit d>")

    def test_cmp(self):
        # should error for incompatible units, if they are compatible then it
        # should cmp on their factors

        x = _find_unit('d')
        y = _find_unit('s')
        z = _find_unit('ft')

        self.assertTrue(x > y)
        self.assertEqual(x, x)
        self.assertTrue(y < x)

        try:
            x < z
        except TypeError as err:
            self.assertEqual(str(err), "Units 'd' and 'ft' are incompatible.")
        else:
            self.fail("Expecting TypeError")

    known__mul__Values = (('1m', '5m', 5), ('1cm', '1cm', 1), ('1cm', '5m', 5),
                          ('7km', '1m', 7))

    def test_multiply(self):
        # multiplication should error for units with offsets

        x = _find_unit('g')
        y = _find_unit('s')
        z = _find_unit('degC')

        self.assertEqual(x * y, PhysicalUnit({'s': 1, 'kg': 1}, .001,
                                             _get_powers(mass=1, time=1), 0))
        self.assertEqual(y * x, PhysicalUnit({'s': 1, 'kg': 1}, .001,
                                             _get_powers(mass=1, time=1), 0))

        try:
            x * z
        except TypeError as err:
            self.assertEqual(
                str(err), "Can't multiply units: either 'g' or 'degC' has a non-zero offset.")
        else:
            self.fail("Expecting TypeError")

    def test_division(self):
        # division should error when working with offset units

        w = _find_unit('kg')
        x = _find_unit('g')
        y = _find_unit('s')
        z = _find_unit('degC')

        quo = w / x
        quo2 = x / y

        self.assertEqual(quo, PhysicalUnit({'kg': 1, 'g': -1},
                                           1000.0, _get_powers(), 0))
        self.assertEqual(quo2, PhysicalUnit({'s': -1, 'g': 1},
                                            0.001,
                                            _get_powers(mass=1, time=-1), 0))
        quo = y / 2.0
        self.assertEqual(quo, PhysicalUnit({'s': 1, "2.0": -1},
                                           .5, _get_powers(time=1), 0))
        quo = 2.0 / y
        self.assertEqual(quo, PhysicalUnit({'s': -1, "2.0": 1}, 2,
                                           _get_powers(time=-1), 0))
        try:
            x / z
        except TypeError as err:
            self.assertEqual(
                str(err), "Can't divide units: either 'g' or 'degC' has a non-zero offset.")
        else:
            self.fail("Expecting TypeError")

    known__pow__Values = (('1V', 3), ('1m', 2), ('1.1m', 2))

    def test_pow(self):
        # power should error for offest units and for non-integer powers

        x = _find_unit('m')
        y = _find_unit('degF')

        z = x**3
        self.assertEqual(z, _find_unit('m**3'))
        x = z**(1.0 / 3.0)  # checks inverse integer units
        self.assertEqual(x, _find_unit('m'))

        # test offset units:
        try:
            y**17
        except TypeError as err:
            self.assertEqual(
                str(err), "Can't exponentiate unit 'degF' because it has a non-zero offset.")
        else:
            self.fail('Expecting TypeError')

        # test non-integer powers
        try:
            x**1.2
        except TypeError as err:
            self.assertEqual(
                str(err), "Can't exponentiate unit 'm': only integer and inverse integer exponents are allowed.")
        else:
            self.fail('Expecting TypeError')
        try:
            x**(5.0 / 2.0)
        except TypeError as err:
            self.assertEqual(
                str(err), "Can't exponentiate unit 'm': only integer and inverse integer exponents are allowed.")
        else:
            self.fail('Expecting TypeError')

    known__conversion_factor_to__Values = (('1m', '1cm', 100), ('1s', '1ms', 1000),
                                           ('1ms', '1s', 0.001))

    def test_conversion_tuple_to(self):
        # test_conversion_tuple_to shoudl error when units have different power
        # lists

        w = _find_unit('cm')
        x = _find_unit('m')
        y = _find_unit('degF')
        z1 = _find_unit('degC')

        # check for non offset units
        self.assertEqual(w.conversion_tuple_to(x), (1 / 100.0, 0))

        # check for offset units
        result = y.conversion_tuple_to(z1)
        self.assertAlmostEqual(result[0], 0.556, 3)
        self.assertAlmostEqual(result[1], -32.0, 3)

        # check for incompatible units
        try:
            x.conversion_tuple_to(z1)
        except TypeError as err:
            self.assertEqual(str(err), "Units 'm' and 'degC' are incompatible.")
        else:
            self.fail("Expecting TypeError")

    def test_name(self):
        # name should return a mathematically correct representation of the
        # unit
        x1 = _find_unit('m')
        x2 = _find_unit('kg')
        y = 1 / x1
        self.assertEqual(y.name(), '1/m')
        y = 1 / x1 / x1
        self.assertEqual(y.name(), '1/m**2')
        y = x1**2
        self.assertEqual(y.name(), 'm**2')
        y = x2 / (x1**2)
        self.assertEqual(y.name(), 'kg/m**2')

    def test_unit_conversion(self):
        self.assertEqual(unit_conversion('km', 'm'), (1000., 0.))

        try:
            unit_conversion('km', 1.0)
        except ValueError as err:
            self.assertEqual(str(err), "The units '1.0' are invalid.")
        else:
            self.fail("Expecting RuntimeError")

    def test_get_conversion(self):
        msg = "'get_conversion' has been deprecated. Use 'unit_conversion' instead."
        with assert_warning(OMDeprecationWarning, msg):
            get_conversion('km', 'm'), (1000., 0.)

        self.assertEqual(get_conversion('km', 'm'), (1000., 0.))

        try:
            get_conversion('km', 1.0)
        except ValueError as err:
            self.assertEqual(str(err), "The units '1.0' are invalid.")
        else:
            self.fail("Expecting ValueError")

    def test_unit_simplification(self):
        test_strings = ['ft/s*s',
                        'm/s*s',
                        'm * ft * cm / km / m',
                        's/s',
                        'm ** 7 / m ** 5']

        correct_strings = ['ft',
                           'm',
                           'ft*cm/km',
                           None,
                           'm**2']

        for test_str, correct_str in zip(test_strings, correct_strings):
            simplified_str = simplify_unit(test_str)
            self.assertEqual(simplified_str, correct_str)

    def test_atto_seconds(self):
        # The unit 'as' was bugged because it is a python keyword.

        fact = unit_conversion('s', 'as')
        assert_near_equal(fact[0], 1e18)

        # Make sure regex for 'as' doesn't pick up partial words.
        fact = unit_conversion('aslug*as*as', 'aslug*zs*zs')
        assert_near_equal(fact[0], 1e6)

        # Make sure simplification works.
        simple = simplify_unit('m*as/as')
        self.assertEqual(simple, 'm')

        simple = simplify_unit('as**6/as**4')
        self.assertEqual(simple, 'as**2')


class TestModuleFunctions(unittest.TestCase):
    def test_add_unit(self):
        try:
            add_unit('ft', '20*m')
        except KeyError as err:
            self.assertEqual(
                err.args[0], "Unit 'ft' already defined with different factor or powers.")
        else:
            self.fail("Expecting Key Error")

        try:
            add_offset_unit('degR', 'degK', 20, 10)
        except KeyError as err:
            self.assertEqual(
                err.args[0], "Unit 'degR' already defined with different factor or powers.")
        else:
            self.fail("Expecting Key Error")

    def test_connect_unitless_to_none(self):
        import warnings
        p = om.Problem()
        ivc = p.model.add_subsystem('indeps', om.IndepVarComp())
        ivc.add_output('x', val=5.0, units='1/s*s')
        ivc.add_output('y', val=10.0, units='Hz*s')
        p.model.add_subsystem('exec_comp', om.ExecComp('z = x + y', z={'units': None},
                                                       x={'units': None}, y={'units': None}))
        p.model.connect('indeps.x', 'exec_comp.x')
        p.model.connect('indeps.y', 'exec_comp.y')

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p.setup()

        p.run_model()
        assert_near_equal(p.get_val('exec_comp.z'), 15.0)

    def test_promote_unitless_and_none(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['x', 'y'])
        ivc.add_output('x', val=5.0, units='1/s*s')
        ivc.add_output('y', val=10.0, units='Hz*s')
        p.model.add_subsystem('exec_comp', om.ExecComp('z = x + y', z={'units': None},
                                                       x={'units': None}, y={'units': None}),
                              promotes_inputs=['x', 'y'])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p.setup()

        p.run_model()
        assert_near_equal(p.get_val('exec_comp.z'), 15.0)

    def test_promote_unitless_ivc_to_exec_comp(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('indeps', om.IndepVarComp())
        ivc.add_output('x', val=5.0, units=None)
        ivc.add_output('y', val=10.0, units='Hz*s')
        p.model.add_subsystem('exec_comp', om.ExecComp('z = x + y', z={'units': None},
                                                       x={'units': '1/s*s'}, y={'units': None}))
        p.model.connect('indeps.x', 'exec_comp.x')
        p.model.connect('indeps.y', 'exec_comp.y')

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p.setup()

        p.run_model()
        assert_near_equal(p.get_val('exec_comp.z'), 15.0)

    def test_incompatible(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['x', 'y'])
        ivc.add_output('x', val=5.0, units='1/s*s')
        ivc.add_output('y', val=10.0, units='Hz*s')
        p.model.add_subsystem('exec_comp', om.ExecComp('z = x + y', z={'units': None},
                                                       x={'units': None}, y={'units': 'ft'}),
                              promotes_inputs=['x', 'y'])

        msg = ("<model> <class Group>: Output units of 'Hz*s' for 'indeps.y' are incompatible with input "
               "units of 'ft' for 'exec_comp.y'.")

        with self.assertRaises(RuntimeError) as cm:
            p.setup()
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
