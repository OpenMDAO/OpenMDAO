"""Test N2 GUI with multiple models using Playwright."""
import unittest
import os

try:
    import playwright
except ImportError:
    class TestOpenMDAOn2GUI(unittest.TestCase):
        def test_n2_gui(self):
            raise unittest.SkipTest("tests require the 'playwright' package.")
else:
    os.system("playwright install")
    from .n2_gui_test import n2_gui_test_case
    from .gen_gui_test import gen_gui_test_case

if __name__ == "__main__":
    unittest.main()
