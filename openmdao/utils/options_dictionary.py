"""Define the OptionsDictionary class."""

from openmdao.utils.general_utils import warn_deprecation


class Undefined(object):
    """
    Class for defining an 'undefined' object which appears in the docs as 'undefined'.
    """

    def __repr__(self):
        """
        Return a string representation for an 'undefined' object.

        Returns
        -------
        str
            'undefined'
        """
        return 'undefined'


# unique object to check if default is given
_undefined = Undefined()


#
# Template for `check_valid` function
#
def check_valid(name, value):
    """
    Check the validity of value for the option with name.

    Parameters
    ----------
    name : str
        name of the option
    value : any
        value for the option

    Raises
    ------
    ValueError
        if value is not valid for option
    """
    raise ValueError(f"Option '{name}' with value {value} is not valid.")


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
    _parent_name : str or None
        If defined, prepend this name to beginning of all exceptions.
    _read_only : bool
        If True, no options can be set after declaration.
    _all_recordable : bool
        Flag to determine if all options in UserOptions are recordable.
    _deprecation_warning_issued : list
        Option names that are deprecated and a warning has been issued for their use.
    """

    def __init__(self, parent_name=None, read_only=False):
        """
        Initialize all attributes.

        Parameters
        ----------
        parent_name : str
            Name or class name of System that owns this OptionsDictionary
        read_only : bool
            If True, setting (via __setitem__ or update) is not permitted.
        """
        self._dict = {}
        self._parent_name = parent_name
        self._read_only = read_only

        self._all_recordable = True

        self._deprecation_warning_issued = []

    def __getstate__(self):
        """
        Return state as a dict.

        Returns
        -------
        dict
            State to get.
        """
        if self._all_recordable:
            return self.__dict__
        else:
            state = self.__dict__.copy()
            state['_dict'] = {key: val for key, val in state['_dict'].items() if val['recordable']}
            return state

    def __repr__(self):
        """
        Return a dictionary representation of the options.

        Returns
        -------
        dict
            The options dictionary.
        """
        return self._dict.__repr__()

    def __rst__(self):
        """
        Generate reStructuredText view of the options table.

        Returns
        -------
        list of str
            A rendition of the options as an rST table.
        """
        outputs = []
        for option_name, option_data in sorted(self._dict.items()):
            name = option_name
            default = option_data['value'] if option_data['value'] is not _undefined \
                else '**Required**'
            values = option_data['values']
            types = option_data['types']
            desc = option_data['desc']

            # if the default is an object instance, replace with the (unqualified) object type
            default_str = str(default)
            idx = default_str.find(' object at ')
            if idx >= 0 and default_str[0] == '<':
                parts = default_str[:idx].split('.')
                default = parts[-1]

            if types is None:
                types = "N/A"

            elif types is not None:
                if not isinstance(types, (set, tuple, list)):
                    types = (types,)

                types = [type_.__name__ for type_ in types]

            if values is None:
                values = "N/A"

            elif values is not None:
                if not isinstance(values, (set, tuple, list)):
                    values = (values,)

                values = [value for value in values]

            outputs.append([name, default, values, types, desc])

        lines = []

        col_heads = ['Option', 'Default', 'Acceptable Values', 'Acceptable Types', 'Description']

        max_sizes = {}
        for j, col in enumerate(col_heads):
            max_sizes[j] = len(col)

        for output in outputs:
            for j, item in enumerate(output):
                length = len(str(item))
                if max_sizes[j] < length:
                    max_sizes[j] = length

        header = ""
        titles = ""
        for key, val in max_sizes.items():
            header += '=' * val + ' '

        for j, head in enumerate(col_heads):
            titles += "%s " % head
            size = max_sizes[j]
            space = size - len(head)
            if space > 0:
                titles += space * ' '

        lines.append(header)
        lines.append(titles)
        lines.append(header)

        n = 3
        for output in outputs:
            line = ""
            for j, item in enumerate(output):
                line += "%s " % str(item)
                size = max_sizes[j]
                space = size - len(str(item))
                if space > 0:
                    line += space * ' '

            lines.append(line)
            n += 1

        lines.append(header)

        return lines

    def __str__(self, width=100):
        """
        Generate text string representation of the options table.

        Parameters
        ----------
        width : int
            The maximum width of the text.

        Returns
        -------
        str
            A text representation of the options table.
        """
        rst = self.__rst__()
        cols = [len(header) for header in rst[0].split()]
        desc_col = sum(cols[:-1]) + len(cols) - 1
        desc_len = width - desc_col

        # if it won't fit in allowed width, just return the rST
        if desc_len < 10:
            return '\n'.join(rst)

        text = []
        for row in rst:
            if len(row) > width:
                text.append(row[:width])
                if not row.startswith('==='):
                    row = row[width:].rstrip()
                    while(len(row) > 0):
                        text.append(' ' * desc_col + row[:desc_len])
                        row = row[desc_len:]
            else:
                text.append(row)

        return '\n'.join(text)

    def _raise(self, msg, exc_type=RuntimeError):
        """
        Raise the given exception type, with parent's name prepended to the message.

        Parameters
        ----------
        msg : str
            The error message.
        exc_type : class
            The type of the exception to be raised.
        """
        if self._parent_name is None:
            full_msg = msg
        else:
            full_msg = '{}: {}'.format(self._parent_name, msg)
        raise exc_type(full_msg)

    def _assert_valid(self, name, value):
        """
        Check whether the given value is valid, where the key has already been declared.

        The optional checks consist of ensuring: the value is one of a list of acceptable values,
        the type of value is one of a list of acceptable types, value is not less than lower,
        value is not greater than upper, and value satisfies check_valid.

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

        if not (value is None and meta['allow_none']):
            # If only values is declared
            if values is not None:
                if value not in values:
                    if isinstance(value, str):
                        value = "'{}'".format(value)
                    self._raise("Value ({}) of option '{}' is not one of {}.".format(value, name,
                                                                                     values),
                                ValueError)
            # If only types is declared
            elif types is not None:
                if not isinstance(value, types):
                    vtype = type(value).__name__

                    if isinstance(value, str):
                        value = "'{}'".format(value)

                    if isinstance(types, (set, tuple, list)):
                        typs = tuple([type_.__name__ for type_ in types])
                        self._raise("Value ({}) of option '{}' has type '{}', but one of "
                                    "types {} was expected.".format(value, name, vtype, typs),
                                    exc_type=TypeError)
                    else:
                        self._raise("Value ({}) of option '{}' has type '{}', but type '{}' "
                                    "was expected.".format(value, name, vtype, types.__name__),
                                    exc_type=TypeError)

            if upper is not None:
                if value > upper:
                    self._raise("Value ({}) of option '{}' "
                                "exceeds maximum allowed value of {}.".format(value, name, upper),
                                exc_type=ValueError)
            if lower is not None:
                if value < lower:
                    self._raise("Value ({}) of option '{}' "
                                "is less than minimum allowed value of {}.".format(value, name,
                                                                                   lower),
                                exc_type=ValueError)

        # General function test
        if meta['check_valid'] is not None:
            meta['check_valid'](name, value)

    def declare(self, name, default=_undefined, values=None, types=None, desc='',
                upper=None, lower=None, check_valid=None, allow_none=False, recordable=True,
                deprecation=None):
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
        desc : str
            Optional description of the option.
        upper : float or None
            Maximum allowable value.
        lower : float or None
            Minimum allowable value.
        check_valid : function or None
            User-supplied function with arguments (name, value) that raises an exception
            if the value is not valid.
        allow_none : bool
            If True, allow None as a value regardless of values or types.
        recordable : bool
            If True, add to recorder
        deprecation : str or None
            If None, it is not deprecated. If a str, use as a DeprecationWarning
            during __setitem__ and __getitem__
        """
        if values is not None and not isinstance(values, (set, list, tuple)):
            self._raise("In declaration of option '%s', the 'values' arg must be of type None,"
                        " list, or tuple - not %s." % (name, values), exc_type=TypeError)

        if types is not None and not isinstance(types, (type, set, list, tuple)):
            self._raise("In declaration of option '%s', the 'types' arg must be None, a type "
                        "or a tuple - not %s." % (name, types), exc_type=TypeError)

        if types is not None and values is not None:
            self._raise("'types' and 'values' were both specified for option '%s'." % name)

        if types is bool:
            values = (True, False)

        if not recordable:
            self._all_recordable = False

        default_provided = default is not _undefined

        if default_provided and default is None:
            # specifying default=None implies allow_none
            allow_none = True

        self._dict[name] = {
            'value': default,
            'values': values,
            'types': types,
            'desc': desc,
            'upper': upper,
            'lower': lower,
            'check_valid': check_valid,
            'has_been_set': default_provided,
            'allow_none': allow_none,
            'recordable': recordable,
            'deprecation': deprecation,
        }

        # If a default is given, check for validity
        if default_provided:
            self._assert_valid(name, default)

    def undeclare(self, name):
        """
        Remove entry from the OptionsDictionary, for classes that don't use that option.

        Parameters
        ----------
        name : str
            The name of a key, the entry of which will be removed from the internal dictionary.

        """
        if name in self._dict:
            del self._dict[name]

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
        try:
            meta = self._dict[name]
        except KeyError:
            # The key must have been declared.
            msg = "Option '{}' cannot be set because it has not been declared."
            self._raise(msg.format(name), exc_type=KeyError)

        if meta['deprecation'] is not None and name not in self._deprecation_warning_issued:
            warn_deprecation(meta['deprecation'])
            self._deprecation_warning_issued.append(name)

        if self._read_only:
            self._raise("Tried to set read-only option '{}'.".format(name), exc_type=KeyError)

        self._assert_valid(name, value)

        meta['value'] = value
        meta['has_been_set'] = True

    def __getitem__(self, name):
        """
        Get an option from the dict or declared default.

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
            if meta['deprecation'] is not None and name not in self._deprecation_warning_issued:
                warn_deprecation(meta['deprecation'])
                self._deprecation_warning_issued.append(name)
            if meta['has_been_set']:
                return meta['value']
            else:
                self._raise("Option '{}' is required but has not been set.".format(name))
        except KeyError:
            self._raise("Option '{}' cannot be found".format(name), exc_type=KeyError)
