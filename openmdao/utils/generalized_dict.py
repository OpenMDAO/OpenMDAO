"""Define the GeneralizedDictionary class."""
from __future__ import division, print_function


class GeneralizedDictionary(object):
    """
    Dictionary with type-checking and default values of declared keys.

    This class is instantiated for:
        1. the metadata attribute in systems.

    Attributes
    ----------
    _dict : dict
        dictionary of entries set using via dictionary access.
    _global_dict : dict
        dictionary of entries like _dict, but combined with dicts of parents.
    _declared_entries : dict
        dictionary of entry declarations.
    """

    def __init__(self):
        """
        Initialize all attributes.

        Parameters
        ----------
        in_dict : dict or None
            optional dictionary with which to initialize.
        """
        self._dict = {}
        self._global_dict = {}
        self._declared_entries = {}

    def _check_type_and_value(self, name, value):
        """
        If declared, check that value has the right type and is valid.

        Parameters
        ----------
        name : str
            the name of the entry, which may or may not have been declared.
        value : -
            the value of the entry to be put in _dict or _global_dict.
        """
        type_ = self._declared_entries[name]['type']
        values = self._declared_entries[name]['values']

        # (1) Check the type
        if type_ is not None and not isinstance(value, type_):
            raise ValueError("Entry '{}' has the wrong type ({})".format(
                name, type_))
        # (2) Check the value
        if values is not None and value not in values:
            raise ValueError("Entry '{}'\'s value is not one of {}".format(
                name, values))

        # Check bounds violation
        upper = self._declared_entries[name]['upper']
        lower = self._declared_entries[name]['lower']
        if upper is not None:
            if value > upper:
                msg = ("Value of {} exceeds maximum of {} for entry 'x'")
                raise ValueError(msg.format(value, upper))
        if lower is not None:
            if value < lower:
                msg = ("Value of {} exceeds minimum of {} for entry 'x'")
                raise ValueError(msg.format(value, lower))

    def declare(self, name, type_=None, desc='', value=None, values=None, required=False,
                upper=None, lower=None):
        """
        Declare an entry.

        Parameters
        ----------
        name : str
            The name of the entry.
        type_ : type or None
            Type of the entry in _dict or _global_dict.
        desc : str
            Description of the entry.
        value : -
            The default value of the entry.
        values : [-, ...]
            The allowed values of the entry.
        required : boolean
            If True, this entry must be specified in _dict or _global_dict.
        upper : float
            Maximum allowable value.
        lower : float
            Minimum allowable value.
        """
        self._declared_entries[name] = {
            'type': type_,
            'desc': desc,
            'value': value,
            'values': values,
            'required': required,
            'upper': upper,
            'lower': lower,
        }

        # If the entry has already been set, check if valid:
        if name in self._dict:
            self._check_type_and_value(name, self._dict[name])

    def update(self, in_dict):
        """
        Update the internal dictionary with the given one.

        Parameters
        ----------
        in_dict : dict
            the incoming dictionary to add to / overwrite the internal one.
        """
        for key in in_dict:
            self[key] = in_dict[key]

    def _assemble_global_dict(self, parents_dict):
        """
        Incorporate the dictionary passed down from the systems above.

        Parameters
        ----------
        parents_dict : dict
            combination of the dict entries of all systems above this one.
        """
        # Reset the _global_dict attribute
        self._global_dict = {}

        # Loop over the passed in dict and insert into _global_dict
        for name in parents_dict:
            value = parents_dict[name]

            # If this is a declared entry:
            if name in self._declared_entries:
                self._check_type_and_value(name, value)

            self._global_dict[name] = value

        # Add the local entries, overwriting when there are conflicts
        self._global_dict.update(self._dict)

    def __iter__(self):
        """
        Provide an iterator.

        Returns
        -------
        iterable
            iterator over the keys in the dictionary.
        """
        return iter(self._dict)

    def __contain__(self, key):
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
        # If this is a declared entry:
        if name in self._declared_entries:
            self._check_type_and_value(name, value)

        self._dict[name] = value

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
            return self._dict[name]

        # If the entry has been set in a system above, return the set value
        if name in self._global_dict:
            return self._global_dict[name]

        # If this is a declared entry:
        if name in self._declared_entries:
            value = self._declared_entries[name]['value']
            required = self._declared_entries[name]['required']

            # If a default value is available:
            if value is not None:
                return value
            # If not, raise an error
            else:
                raise KeyError("Entry '{}' has no default".format(name))

        raise KeyError("Entry '{}' cannot be found".format(name))


class OptionsDictionary(GeneralizedDictionary):
    """
    Dictionary with enforced type-checking and default values of declared keys.

    This class is instantiated for:
        1. the options attribute in solvers, drivers, and processor allocators
        2. the supports attribute in drivers

    Attributes
    ----------
    _dict : dict
        Dictionary of entries set using via dictionary access.
    _global_dict : dict
        Dictionary of entries like _dict, but combined with dicts of parents.
    _declared_entries : dict
        Dictionary of entry declarations.
    _read_only : bool
        Flag that toggles read_only mode.
    """

    def __init__(self, read_only=False):
        """
        Initialize all attributes.

        Parameters
        ----------
        read_only : bool
            Set to True to create a read-only OptionsDictionary.
        """
        super(OptionsDictionary, self).__init__()

        self._read_only = read_only

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

        if name not in self._declared_entries:
            raise KeyError("Entry '{}' is not declared".format(name))

        self._check_type_and_value(name, value)
        self._dict[name] = value

    def _assemble_global_dict(self, parents_dict):
        """
        Incorporate the dictionary passed down from the systems above.

        Parameters
        ----------
        parents_dict : dict
            combination of the dict entries of all systems above this one.
        """
        # Reset the _global_dict attribute
        self._global_dict = {}

        # Loop over the passed in dict and insert into _global_dict
        for name in parents_dict:
            value = parents_dict[name]
            if name not in self._declared_entries:
                raise KeyError("Entry '{}' is not declared".format(name))
            self._check_type_and_value(name, value)
            self._global_dict[name] = value

        # Add the local entries, overwriting when there are conflicts
        self._global_dict.update(self._dict)
