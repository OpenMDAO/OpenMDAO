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

from openmdao.utils.om_warnings import issue_warning

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


def get_code(reference, hide_doc_string=False):
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
    IPython.display.Code
        Source code of the given class or function.
    """
    obj = inspect.getsource(_get_object_from_reference(reference))

    if hide_doc_string:
        obj = obj.split('"""')
        del obj[1]
        obj = ''.join(obj)

    if ipy:
        return Code(obj, language='python')
    else:
        issue_warning("IPython is not installed. Run `pip install openmdao[notebooks]` or "
                      "`pip install openmdao[docs]` to upgrade.")


def display_source(reference, hide_doc_string=False):
    """
    Display the source code of the given reference path to a function.

    Parameters
    ----------
    reference : str
        Dot path of desired function.
    hide_doc_string : bool
        Option to hide the docstring.
    """
    if ipy:
        display(get_code(reference, hide_doc_string))


def show_options_table(reference, recording_options=False):
    """
    Return the options table of the given reference path.

    Parameters
    ----------
    reference : str or object
        Dot path of desired class or function or an instance.

    recording_options : bool
        If True, display recording options instead of options.

    Returns
    -------
    IPython.display
        Options table of the given class or function.
    """
    if isinstance(reference, str):
        obj = _get_object_from_reference(reference)()
    else:
        obj = reference

    if ipy:
        if not hasattr(obj, "options"):
            html = obj.to_table(fmt='html')
        elif not recording_options:
            html = obj.options.to_table(fmt='html')
        else:
            html = obj.recording_options.to_table(fmt='html')

        # Jupyter notebook imposes right justification, so we have to enforce what we want:
        # - Center table headers
        # - Left justify table columns
        # - Limit column width so there is adequate width left for the deprecation message
        style = '<{tag} style="text-align:{align}; max-width:{width}; overflow-wrap:break-word;">'

        cols = html.count('<th>')                 # there could be 5 or 6 columns
        width = '300px' if cols > 5 else '600px'  # limit width depending on number of columns

        html = html.replace('<th>', style.format(tag='th', align='center', width=width))
        html = html.replace('<td>', style.format(tag='td', align='left', width=width))

        return display(HTML(html))
    else:
        issue_warning("IPython is not installed. Run `pip install openmdao[notebooks]` or "
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
        issue_warning("Tabulate is not installed. Run `pip install openmdao[notebooks]` to "
                      "install required dependencies. Using ASCII for outputs.")
    return ipy


notebook = notebook_mode()
