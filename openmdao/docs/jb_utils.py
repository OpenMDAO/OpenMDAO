import importlib
import inspect
from IPython.display import display, HTML, Code


def get_object_from_reference(reference):
    split = reference.split('.')
    right = []
    module = None
    while split:
        try:
            module = importlib.import_module('.'.join(split))
            break
        except ModuleNotFoundError:
            right.append(split.pop())
    if module:
        for entry in reversed(right):
            module = getattr(module, entry)
    return module

def display_source(reference, hide_doc_string=False):
    """
    Return the source code of the given reference path.

    Parameters
    ----------
    reference : str
        Dot path of desired class or function.

    Returns
    -------
    str
        Source code of the given class or function.
    """
    obj = inspect.getsource(get_object_from_reference(reference))

    if hide_doc_string:
        obj = obj.split('"""')
        del obj[1]
        obj = ''.join(obj)

    return Code(obj, language='python')

def show_options_table(reference):
    """
    Return the options table of the given reference path.

    Parameters
    ----------
    reference : str
        Dot path of desired class or function.

    Returns
    -------
    IPython.display.Code
        Options table of the given class or function.
    """
    obj = get_object_from_reference(reference)()

    return display(HTML(obj.options.to_table(fmt='html')))

def cite(reference):
    """
    Return the citation of the given reference path.

    Parameters
    ----------
    reference : str
        Dot path of desired class or function.
    """
    obj = get_object_from_reference(reference)()

    print(obj.cite)

    return