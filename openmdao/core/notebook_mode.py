"""Checking for interactive notebook mode."""


def _notebook_mode():

    try:
        from IPython import get_ipython
        ipy = get_ipython() is not None
    except ImportError:
        pass
    else:
        try:
            import tabulate
        except ImportError:
            print("Tabulate is not installed run `pip install openmdao[notebooks]` to install "
                  "required dependencies")
    return ipy


if _notebook_mode():
    notebook = True
else:
    notebook = False
