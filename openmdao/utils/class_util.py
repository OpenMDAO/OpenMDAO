"""Various utils dealing with classes."""

import weakref


def overrides_method(method_name, obj, base):
    """
    Return True if the named base class method is overridden by obj.

    Note that this only works if the overriding method is declared as part of the class, i.e.,
    if the overriding method is added to the object instance dynamically, it will not be detected
    and this function will return False.

    Parameters
    ----------
    method_name : str
        Name of the method to search for.
    obj : object
        An object that is assumed to inherit from base.
    base : class
        The base class that contains the base version of the named
        method.

    Returns
    -------
    bool
        True if the named base class method is overridden by obj's class or some class in its
        class' mro, otherwise False.
    """
    for klass in obj.__class__.__mro__:
        if method_name in klass.__dict__:
            return klass is not base

    return False


class WeakMethodWrapper(object):
    """
    A class to contain a weak ref to a method.

    weakerf.ref(obj.method) doesn't work, so this class will wrap a weak ref
    to the method's parent object, look the method up in that instance by name, and call it.

    Parameters
    ----------
    obj : object
        The instance object.
    fname : str
        The name of the method.

    Attributes
    ----------
    _ref : weakerf
        The weakref to the method's owning instance.
    __name__ : str
        The name of the method.
    """

    def __init__(self, obj, fname):
        """
        Initialize the wrapper.
        """
        self._ref = weakref.ref(obj)
        self.__name__ = fname

    def __call__(self, *args, **kwargs):
        """
        Call the named method on the object.

        Parameters
        ----------
        *args : tuple of obj
            Positional args.
        **kwargs : dict
            Named args.

        Returns
        -------
        object
            The return value of the wrapped method called with the given args.
        """
        return getattr(self._ref(), self.__name__)(*args, **kwargs)


class _ForwardDescriptor:
    """
    Descriptor that forwards attribute access to a wrapped object.

    Parameters
    ----------
    wrapped_attr : str
        Name of the attribute in the wrapping class that holds the wrapped instance.
    attr_name : str
        Name of the attribute to forward.
    """
    def __init__(self, wrapped_attr, attr_name):
        self.wrapped_attr = wrapped_attr
        self.attr_name = attr_name

    def __get__(self, instance, objtype=None):
        if instance is None:  # class level access
            return self
        return getattr(getattr(instance, self.wrapped_attr), self.attr_name)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.wrapped_attr), self.attr_name, value)

    def __delete__(self, instance):
        delattr(getattr(instance, self.wrapped_attr), self.attr_name)


def auto_forward(wrapped_attr, attr_names):
    """
    Class decorator that automatically forwards attributes from the wrapping class
    to the wrapped class using descriptors.

    Parameters
    ----------
    wrapped_attr : str
        Name of the attribute in the wrapping class that holds the wrapped instance.
    attr_names : list of str
        List of attribute names to forward from the wrapping class to the wrapped instance.

    Returns
    -------
    function
        The class decorator function.

    Examples
    --------
    >>> class Wrapped:
    ...     def __init__(self):
    ...         self.x = 1
    ...         self.y = 2
    ...
    >>> @auto_forward('wrapped_obj', ['x', 'y'])
    ... class Wrapper:
    ...     def __init__(self):
    ...         self.wrapped_obj = Wrapped()
    ...
    >>> w = Wrapper()
    >>> w.x  # Accesses w.wrapped_obj.x
    1
    >>> w.y  # Accesses w.wrapped_obj.y
    2
    """
    def decorator(cls):
        for attr_name in attr_names:
            if attr_name in cls.__dict__:
                raise TypeError(f"Class {cls.__name__} already has an attribute named {attr_name}.")
            setattr(cls, attr_name, _ForwardDescriptor(wrapped_attr, attr_name))
        return cls
    return decorator

