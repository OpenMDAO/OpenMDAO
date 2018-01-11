"""Various utils dealing with classes."""


def overrides_method(method_name, obj, base):
    """
    Return True if the named base class method is overridden by obj.

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
        True if the named base clas is overridden by the given obj, otherwise
        False.
    """
    for klass in obj.__class__.__mro__:
        if method_name in klass.__dict__:
            return klass is not base

    return False
