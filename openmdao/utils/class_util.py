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
