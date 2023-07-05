"""Test Jupyter doc GUI mods specific to OpenMDAO using Playwright."""
import unittest
import os

try:
    import playwright
    import aiounittest
except ImportError:
    class TestOpenMDAOJupyterBookDocs(unittest.TestCase):
        def test_jupyter_book_docs(self):
            raise unittest.SkipTest("tests require the 'playwright' and 'aiounittest' packages.")
else:
    os.system("playwright install")
    from .jupyter_gui_test import TestOpenMDAOJupyterBookDocs


if __name__ == "__main__":
    unittest.main()
