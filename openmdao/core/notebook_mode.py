"""Checking for interactive notebook mode."""
from openmdao.utils.general_utils import simple_warning


def notebook_mode():
    ipy = False
    try:
        from IPython import get_ipython
        ipy = get_ipython() is not None
    except ImportError:
        pass

    if ipy:
        try:
            import tabulate
        except ImportError:
            simple_warning("Tabulate is not installed run `pip install openmdao[notebooks]` to "
                           "install required dependencies. Using ASCII for outputs.")
    return ipy

notebook = notebook_mode()