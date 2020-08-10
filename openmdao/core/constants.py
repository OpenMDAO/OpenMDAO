"""File to define different setup status constants."""

from enum import IntEnum


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
