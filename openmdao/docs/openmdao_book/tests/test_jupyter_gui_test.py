"""Test Jupyter doc GUI mods specific to OpenMDAO using Playwright."""
import unittest
import os

os.system("playwright install")
from .jupyter_gui_test import TestOpenMDAOJupyterBookDocs

if __name__ == "__main__":
    unittest.main()
