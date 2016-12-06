"""Define the units library."""

from __future__ import division, print_function


# Temporary implementation of the units library
def get_units(units):
    """Process the given unit type.

    Args
    ----
    units : str
        String representation of the units.

    Returns
    -------
    float
        Offset to get to default unit: m (length), s(time), etc.
    float
        Mult. factor to get to default unit: m (length), s(time), etc.
    """
    units_dict = {}
    units_dict[''] = (0., 1.)

    # 0: length
    units_dict['m'] = (0., 1.0e0)
    units_dict['mm'] = (0., 1.0e-3)
    units_dict['km'] = (0., 1.0e3)
    units_dict['cm'] = (0., 1.0e-2)

    return units_dict[units]


def is_compatible(old_units, new_units):
    """Check whether units are compatible in terms of base units.

    e.g., m/s is compatible with ft/hr

    Args
    ----
    old_units : str
        original units as a string.
    new_units : str or None
        new units to return the value in; if None, return in standard units.

    Returns
    -------
    bool
        whether the units are compatible.
    """
    return True


def convert_units(val, old_units, new_units=None):
    """Take a given quantity and return in different units.

    Args
    ----
    val : float
        value in original units.
    old_units : str
        original units as a string.
    new_units : str or None
        new units to return the value in; if None, return in standard units.

    Returns
    -------
    float
        value in new units.
    """
    old_c0, old_c1 = get_units(old_units)
    val_std = old_c0 + old_c1 * val
    if new_units is None:
        return val_std
    else:
        new_c0, new_c1 = get_units(new_units)
        # if old_unit_type is not -1 and new_unit_type is not -1:
        assert is_compatible(old_units, new_units)
        return -new_c0 / new_c1 + val_std / new_c1


if __name__ == '__main__':
    for returned, expected in [
        (get_units('cm'), (0., 1.0e-2)),
        (get_units('km'), (0., 1.0e3)),
        (convert_units(3.0, 'mm'), (3.0e-3)),
        (convert_units(3.0, 'mm', 'cm'), (3.0e-1))
    ]:
        print(returned, 'should be', expected)
