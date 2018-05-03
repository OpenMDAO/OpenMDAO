"""Define the OptionsDictionary class."""
from __future__ import division, print_function

from openmdao.utils.general_utils import warn_deprecation

# unique object to check if default is given
_undefined = object()


class OptionsDictionary(object):
    """
    Dictionary with pre-declaration of keys for value-checking and default values.

    This class is instantiated for:
        1. the options attribute in solvers, drivers, and processor allocators
        2. the supports attribute in drivers
        3. the options attribute in systems

    Attributes
    ----------
    _dict : dict of dict
        Dictionary of entries. Each entry is a dictionary consisting of value, values,
        types, desc, lower, and upper.
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
            The key for the declared option.
        value : object
            The default or user-set value to check for value, type, lower, and upper.
        """
        meta = self._dict[name]
        values = meta['values']
        types = meta['types']
        lower = meta['lower']
        upper = meta['upper']
        is_valid = meta['is_valid']

        if not (value is None and meta['allow_none']):
            # If only values is declared
            if values is not None:
                if value not in values:
                    raise ValueError("Option '{}'\'s value is not one of {}".format(name, values))
            # If only types is declared
            elif types is not None:
                if not isinstance(value, types):
                    raise TypeError("Option '{}' has the wrong type ({})".format(name, types))

            if upper is not None:
                if value > upper:
                    msg = ("Value of {} exceeds maximum of {} for option 'x'")
                    raise ValueError(msg.format(value, upper))
            if lower is not None:
                if value < lower:
                    msg = ("Value of {} exceeds minimum of {} for option 'x'")
                    raise ValueError(msg.format(value, lower))

        # General function test
        if is_valid is not None and not is_valid(value):
            raise ValueError("Function is_valid returns False for {}.".format(name))

    def declare(self, name, default=_undefined, values=None, types=None, type_=None, desc='',
                upper=None, lower=None, is_valid=None, allow_none=False):
        r"""
        Declare an option.

        The value of the option must satisfy the following:
        1. If values only was given when declaring, value must be in values.
        2. If types only was given when declaring, value must satisfy isinstance(value, types).
        3. It is an error if both values and types are given.

        Parameters
        ----------
        name : str
            Name of the option.
        default : object or Null
            Optional default value that must be valid under the above 3 conditions.
        values : set or list or tuple or None
            Optional list of acceptable option values.
        types : type or tuple of types or None
            Optional type or list of acceptable option types.
        type_ : type or tuple of types or None
            Deprecated.  Use types instead.
        desc : str
            Optional description of the option.
        upper : float or None
            Maximum allowable value.
        lower : float or None
            Minimum allowable value.
        is_valid : function or None
            General check function that returns True if valid.
        allow_none : bool
            If True, allow None as a value regardless of values or types.
        """
        if type_ is not None:
            warn_deprecation("In declaration of option '%s' the '_type' arg is deprecated.  "
                             "Use 'types' instead." % name)
        if types is None:
            types = type_

        if values is not None and not isinstance(values, (set, list, tuple)):
            raise TypeError("'values' must be of type None, list, or tuple - not %s." % values)
        if types is not None and not isinstance(types, (type, set, list, tuple)):
            raise TypeError("'types' must be None, a type or a tuple  - not %s." % types)

        if types is not None and values is not None:
            raise RuntimeError("'types' and 'values' were both specified for option '%s'." %
                               name)

        default_provided = default is not _undefined

        self._dict[name] = {
            'value': default,
            'values': values,
            'types': types,
            'desc': desc,
            'upper': upper,
            'lower': lower,
            'is_valid': is_valid,
            'has_been_set': default_provided,
            'allow_none': allow_none,
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
            name of the option.

        Returns
        -------
        boolean
            whether key is in the local dict.
        """
        return key in self._dict

    def __setitem__(self, name, value):
        """
        Set an option in the local dictionary.

        Parameters
        ----------
        name : str
            name of the option.
        value : -
            value of the option to be value- and type-checked if declared.
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
        Get an option from the local dict, global dict, or declared default.

        Parameters
        ----------
        name : str
            name of the option.

        Returns
        -------
        value : -
            value of the option.
        """
        # If the option has been set in this system, return the set value
        try:
            meta = self._dict[name]
            if meta['has_been_set']:
                return meta['value']
            else:
                raise RuntimeError("Option '{}' is required but has not been set.".format(name))
        except KeyError:
            raise KeyError("Option '{}' cannot be found".format(name))
