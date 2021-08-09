"""Test Jupyter doc GUI mods specific to OpenMDAO using Playwright."""
import unittest
import os
import sys

# Playwright requires Python 3.7 or higher
if sys.version_info >= (3, 7):
    os.system("playwright install")
    from .jupyter_gui_test import TestOpenMDAOJupyterBookDocs

    if __name__ == "__main__":
        unittest.main()
