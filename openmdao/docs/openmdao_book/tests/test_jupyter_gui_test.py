"""Test Jupyter doc GUI mods specific to OpenMDAO using Playwright."""
import unittest
import os

try:
    import playwright
except ImportError:
    playwright = None

if playwright:
    os.system("playwright install")
    from .jupyter_gui_test import TestOpenMDAOJupyterBookDocs

    if __name__ == "__main__":
        unittest.main()
