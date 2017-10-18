"""Define the OptionsDictionary class."""
from __future__ import division, print_function


class Null(object):
    """
    Dummy class instantiated to check whether the default argument is given.
    """

    pass


null_object = Null()


class OptionsDictionary(object):
    """
    Dictionary with pre-declaration of keys for value-checking and default values.

    This class is instantiated for:
        1. the options attribute in solvers, drivers, and processor allocators
        2. the supports attribute in drivers
        3. the metadata attribute in systems

    Attributes
    ----------
    _dict : dict of dict
        Dictionary of entries. Each entry is a dictionary consisting of value, values,
        type_, desc, lower, and upper.
    _read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """

    def __init__(self, read_only=False):
        """
        Initialize all attributes.

        Parameters
        ----------
        read_only : bool
            If True, setting (via __setitem__ or update) is not permitted.
        """
        self._dict = {}
        self._read_only = read_only

    def _assert_valid(self, name, value):
        """
        Check whether the given value is valid, where the key has already been declared.

        The optional checks consist of ensuring: the value is one of a list of acceptable values,
        the type of value is one of a list of acceptable types, value is not less than lower,
        value is not greater than upper, and value satisfies is_valid.

        Parameters
        ----------
        name : str
            The key for the declared entry.
        value : object
            The default or user-set value to check for value, type, lower, and upper.
        """
        values = self._dict[name]['values']
        type_ = self._dict[name]['type_']
        lower = self._dict[name]['lower']
        upper = self._dict[name]['upper']
        is_valid = self._dict[name]['is_valid']

        # If values and type_ are both declared
        if values is not None and type_ is not None:
            if value not in values and not isinstance(value, type_):
                raise ValueError(
                    "Entry '{}'\'s value is not one of {}".format(name, values)
                    + " and entry '{}' has the wrong type ({})".format(name, type_))
        # If only values is declared
        elif values is not None:
            if value not in values:
                raise ValueError("Entry '{}'\'s value is not one of {}".format(name, values))
        # If only type_ is declared
        elif type_ is not None:
            if not isinstance(value, type_):
                raise TypeError("Entry '{}' has the wrong type ({})".format(name, type_))

        if upper is not None:
            if value > upper:
                msg = ("Value of {} exceeds maximum of {} for entry 'x'")
                raise ValueError(msg.format(value, upper))
        if lower is not None:
            if value < lower:
                msg = ("Value of {} exceeds minimum of {} for entry 'x'")
                raise ValueError(msg.format(value, lower))

        # General function test
        if is_valid is not None and not is_valid(value):
            raise ValueError("Function is_valid returns False for {}.".format(name))

    def declare(self, name, default=null_object, values=None, type_=None, desc='',
                upper=None, lower=None, is_valid=None):
        r"""
        Declare an option.

        The value of the option must satisfy the following:
        1. If values and not type was given when declaring, value must be in values.
        2. If type and not values was given when declaring, value must be an instance of type.
        3. If values and type were given when declaring, either of the above must be true.

        Parameters
        ----------
        name : str
            Name of the option.
        default : object or Null
            Optional default value that must be valid under the above 3 conditions.
        values : set or list or tuple or None
            Optional list of acceptable option values.
        type_ : type or set/list/tuple of types or None
            Optional type or list of acceptable option types.
        desc : str
            Optional description of the option.
        upper : float or None
            Maximum allowable value.
        lower : float or None
            Minimum allowable value.
        is_valid : function or None
            General check function that returns True if valid.
        """
        if values is not None and not isinstance(values, (set, list, tuple)):
            raise TypeError("'values' must be of type None, list, or tuple - not %s." % values)
        if type_ is not None and not isinstance(type_, (type, set, list, tuple)):
            raise TypeError("'type_' must be None, a type or a tuple  - not %s." % type_)

        default_provided = default != null_object

        self._dict[name] = {
            'value': default,
            'values': values,
            'type_': type_,
            'desc': desc,
            'upper': upper,
            'lower': lower,
            'is_valid': is_valid,
            'has_been_set': default_provided,
        }

        # If a default is given, check for validity
        if default_provided:
            self._assert_valid(name, default)

    def update(self, in_dict):
        """
        Update the internal dictionary with the given one.

        Parameters
        ----------
        in_dict : dict
            The incoming dictionary to add to the internal one.
        """
        for name in in_dict:
            self[name] = in_dict[name]

    def __iter__(self):
        """
        Provide an iterator.

        Returns
        -------
        iterable
            iterator over the keys in the dictionary.
        """
        return iter(self._dict)

    def __contains__(self, key):
        """
        Check if the key is in the local dictionary.

        Parameters
        ----------
        key : str
            name of the entry.

        Returns
        -------
        boolean
            whether key is in the local dict.
        """
        return key in self._dict

    def __setitem__(self, name, value):
        """
        Set an entry in the local dictionary.

        Parameters
        ----------
        name : str
            name of the entry.
        value : -
            value of the entry to be value- and type-checked if declared.
        """
        if self._read_only:
            msg = "Tried to set '{}' on a read-only OptionsDictionary."
            raise KeyError(msg.format(name))

        # The key must have been declared.
        if name not in self._dict:
            msg = "Key '{}' cannot be set because it has not been declared."
            raise KeyError(msg.format(name))

        self._assert_valid(name, value)
        self._dict[name]['value'] = value
        self._dict[name]['has_been_set'] = True

    def __getitem__(self, name):
        """
        Get an entry from the local dict, global dict, or declared default.

        Parameters
        ----------
        name : str
            name of the entry.

        Returns
        -------
        value : -
            value of the entry.
        """
        # If the entry has been set in this system, return the set value
        if name in self._dict:
            if self._dict[name]['has_been_set']:
                return self._dict[name]['value']
            else:
                raise RuntimeError("Entry '{}' is required but has not been set.".format(name))
        else:
            raise KeyError("Entry '{}' cannot be found".format(name))
