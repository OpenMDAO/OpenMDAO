"""Checking for interactive notebook mode."""
import sys
import importlib
import inspect

# Fast path: if IPython already loaded, check it
if 'IPython' in sys.modules:
    try:
        import IPython
        ipy = IPython.get_ipython() is not None
    except (ImportError, AttributeError):
        ipy = False
else:
    # IPython not loaded = definitely not in IPython session
    ipy = False

from openmdao.utils.om_warnings import issue_warning, warn_deprecation

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
        from IPython.display import Code
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
    from IPython.display import display
    if ipy:
        display(get_code(reference, hide_doc_string))


def show_options_table(reference, recording_options=False, options_dict='options'):
    """
    Return the options table of the given reference path.

    Parameters
    ----------
    reference : str or object
        Dot path of desired class or function or an instance.
    recording_options : bool
        If True, display recording options instead of options.
    options_dict : str
        The string name of the attribute of the reference object
        that provides to OptionsDictionary.

    Returns
    -------
    IPython.display
        Options table of the given class or function.
    """
    from openmdao.utils.options_dictionary import OptionsDictionary

    if isinstance(reference, str):
        obj = _get_object_from_reference(reference)()
    else:
        obj = reference

    if ipy:
        from IPython.display import display, HTML
        if recording_options:
            warn_deprecation('Argument `recording_options` is deprecated. Use '
                             '`options_dict="recording_options" to remove this '
                             'warning.')
            opt = obj.recording_options
        elif isinstance(obj, OptionsDictionary):
            opt = obj
        elif hasattr(obj, options_dict):
            opt = getattr(obj, options_dict)
        else:
            raise AttributeError(f'Object {reference} has no attribute {options_dict}.')

        return display(HTML(str(opt.to_table(fmt='html', display=False))))
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


def mpi_exec(nprocs, script):
    """
    Run a Python script under mpiexec and display output in the notebook.

    Displays an MPI warning admonition, the syntax-highlighted source of the
    script, the command being run, captured stdout, and raises RuntimeError if
    the script exits with a non-zero return code.

    Parameters
    ----------
    nprocs : int
        Number of MPI processes to use.
    script : str
        Path to the Python script to run under mpiexec.
    """
    import subprocess
    from pathlib import Path

    source = Path(script).read_text(encoding='utf-8')

    if ipy:
        from IPython.display import display, HTML

        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import HtmlFormatter
            formatter = HtmlFormatter()
            highlighted = highlight(source, PythonLexer(), formatter)
            style_tag = f'<style>{formatter.get_style_defs(".highlight")}</style>'
            source_html = style_tag + highlighted
        except ImportError:
            source_html = f'<pre><code>{source}</code></pre>'

        display(HTML(
            '<div class="admonition note"><p class="admonition-title">Note</p>'
            '<p>This feature requires MPI, and may not be able to be run on Colab or Binder.</p>'
            '</div>'
        ))
        display(HTML(
            f'<p>The following script is executed using '
            f'<code>mpiexec -n {nprocs} {script}</code>:</p>'
            + source_html
        ))
    else:
        print('Note: This feature requires MPI, and may not be able to be run on Colab or Binder.')
        print(f'The following script is executed using mpiexec -n {nprocs} {script}:')
        print(source)

    result = subprocess.run(['mpiexec', '-n', str(nprocs), 'python', script],
                            capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(
            f'mpiexec -n {nprocs} {script} failed with exit code {result.returncode}')


def notebook_mode():
    """
    Check if the environment is interactive.

    Returns
    -------
    bool
        True if the environment is an interactive notebook.
    """
    return ipy


notebook = notebook_mode()
