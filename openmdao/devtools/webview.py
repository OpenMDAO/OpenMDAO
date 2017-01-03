import sys
import os
import webbrowser

def webview(outfile):
    """Pop up a web browser for the given file"""
    if sys.platform == 'darwin':
        os.system('open %s' % outfile)
    else:
        webbrowser.get().open(outfile)

def webview_argv():
    """This is tied to a console script called webview.  It just provides
    a convenient way to pop up a browser to view a specified html file(s).
    """
    for name in sys.argv[1:]:
        if os.path.isfile(name):
            webview(name)
