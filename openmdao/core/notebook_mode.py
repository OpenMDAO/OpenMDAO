"""Checking for interactive notebook mode."""
import sys

from openmdao.utils.general_utils import simple_warning

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

colab = 'google.colab' in sys.modules


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
        from IPython import get_ipython
        ipy = get_ipython() is not None
    except ImportError:
        pass

    if ipy and tabulate is None:
        simple_warning("Tabulate is not installed run `pip install openmdao[notebooks]` to "
                       "install required dependencies. Using ASCII for outputs.")
    return ipy


notebook = notebook_mode()
