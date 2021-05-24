"""
Classes and functions to support unit conversion.

The module provides a basic set of predefined physical quantities
in its built-in library; however, it also supports generation of
personal libararies which can be saved and reused.
This module is based on the PhysicalQuantities module
in Scientific Python, by Konrad Hinsen. Modifications by
Justin Gray.
"""

import sys

import re
import os.path
from collections import OrderedDict

from configparser import RawConfigParser as ConfigParser
from openmdao.warnings import warn_deprecation

# pylint: disable=E0611, F0401
from math import floor, pi

import numpy as np


####################################
# Class Definitions
####################################


class NumberDict(OrderedDict):
    """
    Dictionary storing numerical values.

    An instance of this class acts like an array of numbers with
    generalized (non-integer) indices. A value of zero is assumed
    for undefined entries. NumberDict instances support addition
    and subtraction with other NumberDict instances, and multiplication
    and division by scalars.
    """

    def __getitem__(self, item):
        """
        Get the item, or 0.

        Parameters
        ----------
        item : key
            key to get the item

        Returns
        -------
        int
            value of the given key
        """
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            return 0

    def __coerce__(self, other):
        """
        Change other dict to NumberDicts.

        Parameters
        ----------
        other : dict
            the dict instance to be coerced

        Returns
        -------
        NumberDict
            new NumberDict with keys/values from original
        """
        if isinstance(other, dict):
            other = NumberDict(other)
        return self, other

    def __add__(self, other):
        """
        Add another NumberDict to myself.

        Parameters
        ----------
        other : NumberDict
            the other NumberDict Instance

        Returns
        -------
        NumberDict
            new NumberDict with self+other values
        """
        sum_dict = NumberDict()
        for k, v in self.items():
            sum_dict[k] = v
        for k, v in other.items():
            sum_dict[k] = sum_dict[k] + v
        return sum_dict

    def __sub__(self, other):
        """
        Add another NumberDict from myself.

        Parameters
        ----------
        other : NumberDict
            the other NumberDict Instance

        Returns
        -------
        NumberDict
            new NumberDict instance, with self-other values
        """
        sum_dict = NumberDict()
        for k, v in self.items():
            sum_dict[k] = v
        for k, v in other.items():
            sum_dict[k] = sum_dict[k] - v
        return sum_dict

    def __rsub__(self, other):
        """
        Add subtract myself from another NumberDict.

        Parameters
        ----------
        other : NumberDict
            the other NumberDict Instance

        Returns
        -------
        NumberDict
            new NumberDict instance, with other-self values
        """
        sum_dict = NumberDict()
        for k, v in other.items():
            sum_dict[k] = v
        for k, v in self.items():
            sum_dict[k] = sum_dict[k] - v
        return sum_dict

    def __mul__(self, other):
        """
        Multiply myself by another NumberDict.

        Parameters
        ----------
        other : NumberDict
            the other NumberDict Instance

        Returns
        -------
        NumberDict
            new NumberDict instance, with other*self values
        """
        new = NumberDict()
        for key, value in self.items():
            new[key] = other * value
        return new

    __rmul__ = __mul__

    def __div__(self, other):
        """
        Divide myself by another NumberDict.

        Parameters
        ----------
        other : int
            value to divide by

        Returns
        -------
        NumberDict
            new NumberDict instance, with self/other values
        """
        new = NumberDict()
        for key, value in self.items():
            new[key] = value / other
        return new

    __truediv__ = __div__  # for python 3

    def __repr__(self):
        """
        Return a string deceleration of myself.

        Parameters
        ----------
        other : NumberDict
            the other NumberDict Instance

        Returns
        -------
        str
            str representation for the creation of this NumberDict
        """
        return repr(dict(self))


class PhysicalUnit(object):
    """
    Physical unit.

    A physical unit is defined by a name (possibly composite), a scaling
    factor, and the exponentials of each of the SI base units that enter into
    it. Units can be multiplied, divided, and raised to integer powers.

    Attributes
    ----------
    _names : dict or str
        A dictionary mapping each name component to its
        associated integer power (e.g., C{{'m': 1, 's': -1}})
        for M{m/s}). As a shorthand, a string may be passed
        which is assigned an implicit power 1.
    _factor : float
        A scaling factor.
    _powers : list of int
        The integer powers for each of the nine base units.
    _offset : float
        An additive offset to the base unit (used only for temperatures)
    """

    def __init__(self, names, factor, powers, offset=0):
        """
        Initialize all attributes.

        Parameters
        ----------
        names : dict or str
            A dictionary mapping each name component to its
            associated integer power (e.g., C{{'m': 1, 's': -1}})
            for M{m/s}). As a shorthand, a string may be passed
            which is assigned an implicit power 1.
        factor : float
            A scaling factor.
        powers : list of int
            The integer powers for each of the nine base units.
        offset : float
            An additive offset to the base unit (used only for temperatures).
        """
        if isinstance(names, str):
            self._names = NumberDict(((names, 1),))
        else:
            self._names = names

        self._factor = float(factor)
        self._offset = float(offset)
        self._powers = powers

    def __repr__(self):
        """
        Get the string representation of this unit.

        Returns
        -------
        str
            str representation of how to instantiate this PhysicalUnit
        """
        return 'PhysicalUnit(%s,%s,%s,%s)' % (self._names, self._factor,
                                              self._powers, self._offset)

    def __str__(self):
        """
        Convert myself to string.

        Returns
        -------
        str
            str representation of a PhysicalUnit
        """
        return '<PhysicalUnit ' + self.name() + '>'

    def __lt__(self, other):
        """
        Compare myself to other.

        Parameters
        ----------
        other : PhysicalUnit
            The other physical unit to be compared to

        Returns
        -------
        bool
            self._factor < other._factor
        """
        if self._powers != other._powers or self._offset != other._offset:
            raise TypeError(f"Units '{self.name()}' and '{other.name()}' are incompatible.")

        return self._factor < other._factor

    def __gt__(self, other):
        """
        Compare myself to other.

        Parameters
        ----------
        other : PhysicalUnit
            The other physical unit to be compared to

        Returns
        -------
        bool
            self._factor > other._factor
        """
        if self._powers != other._powers:
            raise TypeError(f"Units '{self.name()}' and '{other.name()}' are incompatible.")
        return self._factor > other._factor

    def __eq__(self, other):
        """
        Test for equality.

        Parameters
        ----------
        other : PhysicalUnit
            The other physical unit to be compared to

        Returns
        -------
        bool
            true if _factor, _offset, and _powers all match
        """
        return (self._factor == other._factor and
                self._offset == other._offset and
                self._powers == other._powers)

    def __mul__(self, other):
        """
        Multiply myself by other.

        Parameters
        ----------
        other : PhysicalUnit
            The other physical unit to be compared to

        Returns
        -------
        PhysicalUnit
            new PhysicalUnit instance representing the product of two units
        """
        if self._offset != 0 or (isinstance(other, PhysicalUnit) and
                                 other._offset != 0):
            raise TypeError(f"Can't multiply units: either '{self.name()}' or '{other.name()}' "
                            "has a non-zero offset.")
        if isinstance(other, PhysicalUnit):
            return PhysicalUnit(self._names + other._names,
                                self._factor * other._factor,
                                [a + b for a, b in zip(self._powers, other._powers)])
        else:
            return PhysicalUnit(self._names + {str(other): 1},
                                self._factor * other,
                                self._powers,
                                self._offset * other)

    __rmul__ = __mul__

    def __div__(self, other):
        """
        Divide myself by other.

        Parameters
        ----------
        other : PhysicalUnit
            The other physical unit to be operated on

        Returns
        -------
        PhysicalUnit
            new PhysicalUnit instance representing the self/other
        """
        if self._offset != 0 or (isinstance(other, PhysicalUnit) and
                                 other._offset != 0):
            raise TypeError(f"Can't divide units: either '{self.name()}' or '{other.name()}' "
                            "has a non-zero offset.")
        if isinstance(other, PhysicalUnit):
            return PhysicalUnit(self._names - other._names,
                                self._factor / other._factor,
                                [a - b for (a, b) in zip(self._powers,
                                                         other._powers)])
        else:
            return PhysicalUnit(self._names + {str(other): -1},
                                self._factor / float(other), self._powers)

    __truediv__ = __div__   # for python 3

    def __rdiv__(self, other):
        """
        Divide other by myself.

        Parameters
        ----------
        other : PhysicalUnit
            The other physical unit to be operated on

        Returns
        -------
        PhysicalUnit
            new PhysicalUnit instance representing the other/self
        """
        return PhysicalUnit({str(other): 1} - self._names,
                            float(other) / self._factor,
                            [-x for x in self._powers])

    __rtruediv__ = __rdiv__

    def __pow__(self, power):
        """
        Raise myself to a power.

        Parameters
        ----------
        power : float or int
            power to raise self by

        Returns
        -------
        PhysicalUnit
            new PhysicalUnit of self^power
        """
        if self._offset != 0:
            raise TypeError(f"Can't exponentiate unit '{self.name()}' because it "
                            "has a non-zero offset.")
        if isinstance(power, int):
            return PhysicalUnit(power * self._names, pow(self._factor, power),
                                [x * power for x in self._powers])
        if isinstance(power, float):
            inv_exp = 1. / power
            rounded = int(floor(inv_exp + 0.5))
            if abs(inv_exp - rounded) < 1.e-10:

                if all([x % rounded == 0 for x in self._powers]):
                    f = self._factor**power
                    p = [x / rounded for x in self._powers]
                    if all([x % rounded == 0 for x in self._names.values()]):
                        names = self._names / rounded
                    else:
                        names = NumberDict()
                        if f != 1.:
                            names[str(f)] = 1
                        for x, name in zip(p, _UNIT_LIB.base_names):
                            names[name] = x
                    return PhysicalUnit(names, f, p)

        raise TypeError(f"Can't exponentiate unit '{self.name()}': "
                        "only integer and inverse integer exponents are allowed.")

    def in_base_units(self):
        """
        Return the base unit equivalent of this unit.

        Returns
        -------
        PhysicalUnit
            the equivalent base unit
        """
        num = ''
        denom = ''
        for unit, power in zip(_UNIT_LIB.base_names, self._powers):
            if power < 0:
                denom = denom + '/' + unit
                if power < -1:
                    denom = denom + '**' + str(-power)
            elif power > 0:
                num = num + '*' + unit
                if power > 1:
                    num = num + '**' + str(power)

        if len(num) == 0:
            num = '1'
        else:
            num = num[1:]

        return _find_unit(num + denom)

    def conversion_tuple_to(self, other):
        """
        Compute the tuple of (factor, offset) for conversion.

        Parameters
        ----------
        other : PhysicalUnit
            Another unit.

        Returns
        -------
        Tuple with two floats
            The conversion factor and offset from this unit to another unit.
        """
        if self._powers != other._powers:
            raise TypeError(f"Units '{self.name()}' and '{other.name()}' are incompatible.")

        # let (s1,d1) be the conversion tuple from 'self' to base units
        #   (ie. (x+d1)*s1 converts a value x from 'self' to base units,
        #   and (x/s1)-d1 converts x from base to 'self' units)
        # and (s2,d2) be the conversion tuple from 'other' to base units
        # then we want to compute the conversion tuple (S,D) from
        #   'self' to 'other' such that (x+D)*S converts x from 'self'
        #   units to 'other' units
        # the formula to convert x from 'self' to 'other' units via the
        #   base units is (by definition of the conversion tuples):
        #     ( ((x+d1)*s1) / s2 ) - d2
        #   = ( (x+d1) * s1/s2) - d2
        #   = ( (x+d1) * s1/s2 ) - (d2*s2/s1) * s1/s2
        #   = ( (x+d1) - (d1*s2/s1) ) * s1/s2
        #   = (x + d1 - d2*s2/s1) * s1/s2
        # thus, D = d1 - d2*s2/s1 and S = s1/s2

        factor = self._factor / other._factor
        offset = self._offset - (other._offset * other._factor / self._factor)
        return (factor, offset)

    def is_compatible(self, other):
        """
        Check for compatibility with another unit.

        Parameters
        ----------
        other : PhysicalUnit
            Another unit.

        Returns
        -------
        bool
            indicates if two units are compatible
        """
        return self._powers == other._powers

    def is_dimensionless(self):
        """
        Dimensionless PQ.

        Returns
        -------
        bool
            indicates if this is dimensionless
        """
        return not any(self._powers)

    def is_angle(self):
        """
        Check if this PQ is an Angle.

        Returns
        -------
        bool
            indicates if this an angle type
        """
        return (self._powers[_UNIT_LIB.base_types['angle']] == 1 and
                sum(self._powers) == 1)

    def set_name(self, name):
        """
        Set the name.

        Parameters
        ----------
        name : str
            the name
        """
        self._names = NumberDict()
        self._names[name] = 1

    def name(self):
        """
        Compute the name of this unit.

        Returns
        -------
        str
            str representation of the unit
        """
        num = ''
        denom = ''
        for unit, power in self._names.items():
            if power < 0:
                denom = denom + '/' + unit
                if power < -1:
                    denom = denom + '**' + str(-power)
            elif power > 0:
                num = num + '*' + unit
                if power > 1:
                    num = num + '**' + str(power)
        if len(num) == 0:
            num = '1'
        else:
            num = num[1:]
        return num + denom


####################################
# Module Functions
####################################

def _new_unit(name, factor, powers):
    """
    Create new Unit.

    Parameters
    ----------
    name : str
        The name of the new unit
    factor : float
        conversion factor to base units
    powers : [int, ...]
        power of base units

    """
    _UNIT_LIB.unit_table[name] = PhysicalUnit(name, factor, powers)


def add_offset_unit(name, baseunit, factor, offset, comment=''):
    """
    Adding Offset Unit.

    Parameters
    ----------
    name : str
        The name of the unit
    baseunit : str or instance of PhysicalUnit
        The unit upon which this offset unit is based.
    factor : str
        The scaling factor used to define the new unit w.r.t. baseunit
    offset : float
        zero offset for new unit
    comment : str
        optional comment to describe unit
    """
    if isinstance(baseunit, str):
        baseunit = _find_unit(baseunit)
    # else, baseunit should be a instance of PhysicalUnit
    # names, factor, powers, offset=0
    unit = PhysicalUnit(baseunit._names, baseunit._factor * factor,
                        baseunit._powers, offset)
    unit.set_name(name)
    if name in _UNIT_LIB.unit_table:
        if (_UNIT_LIB.unit_table[name]._factor != unit._factor or
                _UNIT_LIB.unit_table[name]._powers != unit._powers):
            raise KeyError(f"Unit '{name}' already defined with different factor or powers.")
    _UNIT_LIB.unit_table[name] = unit
    _UNIT_LIB.set('units', name, unit)
    if comment:
        _UNIT_LIB.help.append((name, comment, unit))


def add_unit(name, unit, comment=''):
    """
    Adding Unit.

    Parameters
    ----------
    name : str
        The name of the unit being added. For example: 'Hz'
    unit : str
        definition of the unit w.r.t. some other unit.  For example: '1/s'
    comment : str
        optional comment to describe unit
    """
    if comment:
        _UNIT_LIB.help.append((name, comment, unit))
    if isinstance(unit, str):
        unit = eval(unit, {'__builtins__': None, 'pi': pi},
                    _UNIT_LIB.unit_table)
    unit.set_name(name)
    if name in _UNIT_LIB.unit_table:
        if (_UNIT_LIB.unit_table[name]._factor != unit._factor or
                _UNIT_LIB.unit_table[name]._powers != unit._powers):
            raise KeyError(f"Unit '{name}' already defined with different factor or powers.")

    _UNIT_LIB.unit_table[name] = unit
    _UNIT_LIB.set('units', name, unit)


_UNIT_LIB = ConfigParser()


def _do_nothing(string):
    """
    Make the ConfigParser case sensitive.

    Defines an optionxform for the units configparser that
    does nothing, resulting in a case-sensitive parser.

    Parameters
    ----------
    string : str
        The string to be transformed for the ConfigParser

    Returns
    -------
    str
        The same string that was given as a parameter.
    """
    return string


_UNIT_LIB.optionxform = _do_nothing


def import_library(libfilepointer):
    """
    Import a units library, replacing any existing definitions.

    Parameters
    ----------
    libfilepointer : file
        new library file to work with

    Returns
    -------
    ConfigParser
        newly updated units library for the module
    """
    global _UNIT_LIB
    global _UNIT_CACHE
    _UNIT_CACHE = {}
    _UNIT_LIB = ConfigParser()
    _UNIT_LIB.optionxform = _do_nothing

    # New in Python 3.2: read_file() replaces readfp().
    if sys.version_info >= (3, 2):
        _UNIT_LIB.read_file(libfilepointer)
    else:
        _UNIT_LIB.readfp(libfilepointer)

    required_base_types = ['length', 'mass', 'time', 'temperature', 'angle']
    _UNIT_LIB.base_names = list()
    # used to is_angle() and other base type checking
    _UNIT_LIB.base_types = dict()
    _UNIT_LIB.unit_table = dict()
    _UNIT_LIB.prefixes = dict()
    _UNIT_LIB.help = list()

    for prefix, factor in _UNIT_LIB.items('prefixes'):
        factor, comma, comment = factor.partition(',')
        _UNIT_LIB.prefixes[prefix] = float(factor)

    base_list = [0] * len(_UNIT_LIB.items('base_units'))

    for i, (unit_type, name) in enumerate(_UNIT_LIB.items('base_units')):
        _UNIT_LIB.base_types[unit_type] = i
        powers = list(base_list)
        powers[i] = 1
        # print '%20s'%unit_type, powers
        # cant use add_unit because no base units exist yet
        _new_unit(name, 1, powers)
        _UNIT_LIB.base_names.append(name)

    # test for required base types
    missing = [utype for utype in required_base_types
               if utype not in _UNIT_LIB.base_types]
    if missing:
        raise ValueError("Not all required base types were present in the config file. missing: "
                         f"{missing}, at least {required_base_types} required.")

    _update_library(_UNIT_LIB)
    return _UNIT_LIB


def update_library(filename):
    """
    Update units in current library from `filename`.

    Parameters
    ----------
    filename : string or file
        Source of units configuration data.
    """
    if isinstance(filename, basestring):
        inp = open(filename, 'rU')
    else:
        inp = filename
    try:
        cfg = ConfigParser()
        cfg.optionxform = _do_nothing

        # New in Python 3.2: read_file() replaces readfp().
        if sys.version_info >= (3, 2):
            cfg.read_file(inp)
        else:
            cfg.readfp(inp)

        _update_library(cfg)
    finally:
        inp.close()


def _update_library(cfg):
    """
    Update library from :class:`ConfigParser` `cfg`.

    Parameters
    ----------
    cfg : ConfigParser
        ConfigParser loaded with unit_lib.ini data
    """
    retry1 = set()
    for name, unit in cfg.items('units'):
        data = [item.strip() for item in unit.split(',')]
        if len(data) == 2:
            unit, comment = data
            try:
                add_unit(name, unit, comment)
            except NameError:
                retry1.add((name, unit, comment))
        elif len(data) == 4:
            factor, baseunit, offset, comment = data
            try:
                add_offset_unit(name, baseunit, float(factor), float(offset),
                                comment)
            except NameError:
                retry1.add((name, baseunit, float(factor), float(offset),
                            comment))
        else:
            raise ValueError(f"Unit '{name}' definition {unit} has invalid format.")
    retry_count = 0
    last_retry_count = -1
    while last_retry_count != retry_count and retry1:
        last_retry_count = retry_count
        retry_count = 0
        retry2 = retry1.copy()
        for data in retry2:
            if len(data) == 3:
                name, unit, comment = data
                try:
                    add_unit(name, unit, comment)
                    retry1.remove(data)
                except NameError:
                    retry_count += 1
            else:
                try:
                    name, factor, baseunit, offset, comment = data
                    add_offset_unit(name, factor, baseunit, offset, comment)
                    retry1.remove(data)
                except NameError:
                    retry_count += 1
    if retry1:
        raise ValueError('The following units were not defined because they'
                         ' could not be resolved as a function of any other'
                         ' defined units: %s.' % [x[0] for x in retry1])


_UNIT_CACHE = {}


def _is_unitless(units):
    if units is None:
        return True
    unit_meta = _find_unit(units)
    return unit_meta is not None and unit_meta.is_dimensionless()


def _find_unit(unit, error=False):
    """
    Find unit helper function.

    Parameters
    ----------
    unit : str
        str representing the desired unit
    error : bool
        If True, raise exception if unit isn't found.

    Returns
    -------
    PhysicalUnit
        The actual unit object
    """
    if isinstance(unit, str):

        # Deal with 'as' for attoseconds
        reg1 = re.compile(r'\bas\b')
        unit = re.sub(reg1, 'as_', unit)

        name = unit.strip()
        try:
            unit = _UNIT_CACHE[name]
        except KeyError:
            try:
                unit = eval(name, {'__builtins__': None}, _UNIT_LIB.unit_table)
            except Exception:

                # This unit might include prefixed units that aren't in the
                # unit_table. We must parse them ALL and add them to the
                # unit_table.

                # First character of a unit is always alphabet or $.
                # Remaining characters may include numbers.
                regex = re.compile('[A-Z,a-z]{1}[A-Z,a-z,0-9]*')

                for item in regex.findall(name):
                    item = re.sub(reg1, 'as_', item)

                    # check if this was a compound unit, so each
                    # substring might be a unit
                    try:
                        eval(item, {'__builtins__': None},
                             _UNIT_LIB.unit_table)

                    except Exception:  # maybe is a prefixed unit then
                        base_unit = item[1:].rstrip('_')

                        # check for single letter prefix before unit
                        if(item[0] in _UNIT_LIB.prefixes and
                           base_unit in _UNIT_LIB.unit_table):
                            add_unit(item, _UNIT_LIB.prefixes[item[0]] *
                                     _UNIT_LIB.unit_table[base_unit])

                        # check for double letter prefix before unit
                        elif(item[0:2] in _UNIT_LIB.prefixes and
                             item[2:] in _UNIT_LIB.unit_table):
                            add_unit(item, _UNIT_LIB.prefixes[item[0:2]] *
                                     _UNIT_LIB.unit_table[item[2:]])

                        # no prefixes found, unknown unit
                        else:
                            if error:
                                raise ValueError(f"The units '{name}' are invalid.")
                            return None

                unit = eval(name, {'__builtins__': None}, _UNIT_LIB.unit_table)

            _UNIT_CACHE[name] = unit
    else:
        name = unit

    if not isinstance(unit, PhysicalUnit):
        if error:
            raise ValueError(f"The units '{name}' are invalid.")
        return None

    return unit


def valid_units(unit):
    """
    Return whether the given units are vaild.

    Parameters
    ----------
    unit : str
        String representation of the units.

    Returns
    -------
    bool
        True for valid, False for invalid.
    """
    return _find_unit(unit) is not None


def conversion_to_base_units(units):
    """
    Get the offset and scaler to convert from given units to base units.

    Parameters
    ----------
    units : str
        String representation of the units.

    Returns
    -------
    float
        Offset to get to default unit: m (length), s(time), etc.
    float
        Mult. factor to get to default unit: m (length), s(time), etc.
    """
    if not units:  # dimensionless
        return 0., 1.
    unit = _find_unit(units, error=True)

    return unit._offset, unit._factor


def is_compatible(old_units, new_units):
    """
    Check whether units are compatible in terms of base units.

    e.g., m/s is compatible with ft/hr

    Parameters
    ----------
    old_units : str
        original units as a string.
    new_units : str or None
        new units to return the value in; if None, return in standard units.

    Returns
    -------
    bool
        whether the units are compatible.
    """
    if not old_units and not new_units:  # dimensionless
        return True

    old_unit = _find_unit(old_units, error=True)
    new_unit = _find_unit(new_units, error=True)

    return old_unit.is_compatible(new_unit)


def unit_conversion(old_units, new_units):
    """
    Return conversion factor and offset between old and new units.

    Parameters
    ----------
    old_units : str
        original units as a string.
    new_units : str
        new units to return the value in.

    Returns
    -------
    (float, float)
        Conversion factor and offset
    """
    return _find_unit(old_units, error=True).conversion_tuple_to(_find_unit(new_units, error=True))


def get_conversion(old_units, new_units):
    """
    Return conversion factor and offset between old and new units (deprecated).

    Parameters
    ----------
    old_units : str
        original units as a string.
    new_units : str
        new units to return the value in.

    Returns
    -------
    (float, float)
        Conversion factor and offset
    """
    warn_deprecation("'get_conversion' has been deprecated. Use "
                     "'unit_conversion' instead.")

    return unit_conversion(old_units, new_units)


def convert_units(val, old_units, new_units=None):
    """
    Take a given quantity and return in different units.

    Parameters
    ----------
    val : float
        value in original units.
    old_units : str or None
        original units as a string or None.
    new_units : str or None
        new units to return the value in or None.

    Returns
    -------
    float
        value in new units.
    """
    if not old_units or not new_units:  # one side has no units
        return val

    old_unit = _find_unit(old_units, error=True)
    if new_units:
        new_unit = _find_unit(new_units, error=True)
    else:
        new_unit = old_unit.in_base_units()

    (factor, offset) = old_unit.conversion_tuple_to(new_unit)
    return (val + offset) * factor


def _has_val_mismatch(units1, val1, units2, val2):
    """
    Return True if values differ after unit conversion or if values differ when units are None.

    Parameters
    ----------
    units1 : str or None
        Units for first value.
    val1 : float or ndarray
        First value.
    units2 : str or None
        Units for second value.
    val2 : float or ndarray
        Second value.
    """
    if units1 != units2:
        if units1 is None or units2 is None:
            return True

        # convert units
        val1 = convert_units(val1, units1, new_units=units2)

    rtol = 1e-10
    val1 = np.asarray(val1)
    val2 = np.asarray(val2)

    norm1 = np.linalg.norm(val1)
    if norm1 == 0.:
        return np.linalg.norm(val2) > rtol
    else:
        return np.linalg.norm(val2 - val1) / norm1 > rtol


def simplify_unit(old_unit_str, msginfo=''):
    """
    Simplify unit string using built-in naming method.

    Unit string 'ft*s/s' becomes 'ft'.

    Parameters
    ----------
    old_unit_str : str
        Unit string to simplify.
    msginfo : str
        A string prepended to the ValueError which is raised if the units are invalid.

    Returns
    -------
    str
        Simplified unit string.
    """
    if old_unit_str is None:
        return None

    found_unit = _find_unit(old_unit_str)
    if found_unit is None:
        _msginfo = f'{msginfo}: ' if msginfo else ''
        raise ValueError(f"{_msginfo}The units '{old_unit_str}' are invalid.")

    new_str = found_unit.name()
    if new_str == '1':
        # Special Case. Unity always becomes None.
        new_str = None

    # Restore units 'as' (attoseconds).
    if new_str:
        reg1 = re.compile(r'\bas_\b')
        new_str = reg1.sub('as', new_str)
    return new_str


# Load in the default unit library
file_path = open(os.path.join(os.path.dirname(__file__), 'unit_library.ini'))
with file_path as default_lib:
    import_library(default_lib)


if __name__ == '__main__':
    for returned, expected in [
        (conversion_to_base_units('cm'), (0., 1.0e-2)),
        (conversion_to_base_units('km'), (0., 1.0e3)),
        (convert_units(3.0, 'mm'), (3.0e-3)),
        (convert_units(3.0, 'mm', 'cm'), (3.0e-1)),
        (convert_units(100, 'degC', 'degF'), (212.))
    ]:
        print(returned, 'should be', expected)
