"""Checking for interactive notebook mode."""
from openmdao.utils.general_utils import simple_warning

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None
    simple_warning("Tabulate is not installed run `pip install openmdao[notebooks]` to "
                   "install required dependencies. Using ASCII for outputs.")


def notebook_mode():
    """
    Check if the environment is interactive and if tabulate is installed.

    Returns
    -------
    bool
        True if the environment is an interactive notebook.
    """
    ipy = False
    try:
        from IPython import get_ipython, HTML, display
        ipy = get_ipython() is not None
    except ImportError:
        pass

    return ipy


notebook = notebook_mode()
