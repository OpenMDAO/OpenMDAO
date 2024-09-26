"""Test Jupyter doc GUI mods specific to OpenMDAO using Playwright."""
import unittest
import os

try:
    import playwright   # noqa: F401
    import aiounittest  # noqa: F401
except ImportError:
    class TestOpenMDAOJupyterBookDocs(unittest.TestCase):
        def test_jupyter_book_docs(self):
            raise unittest.SkipTest("tests require the 'playwright' and 'aiounittest' packages.")
else:
    os.system("playwright install")
    from .jupyter_gui_test import TestOpenMDAOJupyterBookDocs  # noqa: F401


if __name__ == "__main__":
    unittest.main()
