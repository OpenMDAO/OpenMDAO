"""Checking for interactive notebook mode."""
import sys
import importlib
import inspect

try:
    from IPython.display import display, HTML, IFrame, Code
    from IPython import get_ipython
    ipy = get_ipython() is not None
except ImportError:
    ipy = display = HTML = IFrame = None

from openmdao.utils.general_utils import simple_warning

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

colab = 'google.colab' in sys.modules


def _get_object_from_reference(reference):
    """
    Return object of given reference path.

    Parameters
    ----------
    reference : str
        Dot path of desired class.

    Returns
    -------
    str
        Object of the given class.
    """
    split = reference.split('.')
    right = []
    obj = None
    while split:
        try:
            obj = importlib.import_module('.'.join(split))
            break
        except ModuleNotFoundError:
            right.append(split.pop())
    if obj:
        for entry in reversed(right):
            obj = getattr(obj, entry)
    return obj


def display_source(reference, hide_doc_string=False):
    """
    Return the source code of the given reference path to a function.

    Parameters
    ----------
    reference : str
        Dot path of desired function.
    hide_doc_string : bool
        Option to hide the docstring.

    Returns
    -------
    IPython.display
        Source code of the given class or function.
    """
    obj = inspect.getsource(_get_object_from_reference(reference))

    if hide_doc_string:
        obj = obj.split('"""')
        del obj[1]
        obj = ''.join(obj)

    if ipy:
        return display(Code(obj, language='python'))
    else:
        simple_warning("IPython is not installed. Run `pip install openmdao[notebooks]` or "
                       "`pip install openmdao[docs]` to upgrade.")


def show_options_table(reference, recording_options=False):
    """
    Return the options table of the given reference path.

    Parameters
    ----------
    reference : str
        Dot path of desired class or function.

    recording_options : bool
        If True, display recording options instead of options.

    Returns
    -------
    IPython.display
        Options table of the given class or function.
    """
    obj = _get_object_from_reference(reference)()

    if ipy:
        if not recording_options:
            return display(HTML(obj.options.to_table(fmt='html')))
        else:
            return display(HTML(obj.recording_options.to_table(fmt='html')))
    else:
        simple_warning("IPython is not installed. Run `pip install openmdao[notebooks]` or "
                       "`pip install openmdao[docs]` to upgrade.")


def cite(reference):
    """
    Return the citation of the given reference path.

    Parameters
    ----------
    reference : str
        Dot path of desired class or function.
    """
    obj = _get_object_from_reference(reference)()

    print(obj.cite)

    return


def notebook_mode():
    """
    Check if the environment is interactive and if tabulate is installed.

    Returns
    -------
    bool
        True if the environment is an interactive notebook.
    """
    if ipy and tabulate is None:
        simple_warning("Tabulate is not installed. Run `pip install openmdao[notebooks]` to "
                       "install required dependencies. Using ASCII for outputs.")
    return ipy


notebook = notebook_mode()
