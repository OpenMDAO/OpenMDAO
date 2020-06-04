"""Define some functions to view Web pages."""
import sys
import os
import webbrowser


def webview(outfile):
    """
    Pop up a web browser for the given file.

    Parameters
    ----------
    outfile : str
        Path to the HTML file to open.

    """
    if sys.platform == 'darwin':
        os.system('open %s' % outfile)
    else:
        webbrowser.get().open(outfile)


def webview_argv():
    """
    Provide a convenient way to pop up a browser to view a specified html file(s).

    This is tied to a console script called webview.
    """
    for name in sys.argv[1:]:
        if os.path.isfile(name):
            webview(name)
