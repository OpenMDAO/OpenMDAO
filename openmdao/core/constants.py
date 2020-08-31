"""
Various objects to be used as constants.
"""

import os
from enum import IntEnum
import numpy as np

# This is the dtype we use for index arrays.  Petsc by default uses 32 bit ints
if os.environ.get('OPENMDAO_USE_BIG_INTS'):
    INT_DTYPE = np.dtype(np.int64)
else:
    INT_DTYPE = np.dtype(np.int32)


class _SetupStatus(IntEnum):
    """
    Class used to define different states of the setup status.

    Attributes
    ----------
    PRE_SETUP : int
        Newly initialized problem or newly added model.
    POST_CONFIGURE : int
        Configure has been called.
    POST_SETUP : int
        The `setup` method has been called, but vectors not initialized.
    POST_FINAL_SETUP : int
        The `final_setup` has been run, everything ready to run.
    """

    PRE_SETUP = 0
    POST_CONFIGURE = 1
    POST_SETUP = 2
    POST_FINAL_SETUP = 3


class _ReprClass(object):
    """
    Class for defining objects with a simple constant string __repr__.

    This is useful for constants used in arg lists when you want them to appear in
    automatically generated source documentation as a certain string instead of python's
    default representation.
    """

    def __init__(self, repr_string):
        """
        Inititialize the __repr__ string.

        Parameters
        ----------
        repr_string : str
            The string to be returned by __repr__
        """
        self._repr_string = repr_string

    def __repr__(self):
        """
        Return our _repr_string.

        Returns
        -------
        str
            Whatever string we were initialized with.
        """
        return self._repr_string


# unique object to check if default is given (when None is an allowed value)
_UNDEFINED = _ReprClass("UNDEFINED")

# Use this as a special value to be able to tell if the caller set a value for the optional
# out_stream argument. We run into problems running testflo if we use a default of sys.stdout.
_DEFAULT_OUT_STREAM = _ReprClass("DEFAULT_OUT_STREAM")
