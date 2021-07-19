"""Test N2 GUI with multiple models using Playwright."""
import unittest
import os
import sys

# Playwright requires Python 3.7 or higher
if sys.version_info >= (3, 7):
    os.system("playwright install")
    from n2_gui_test import n2_gui_test_case

    if __name__ == "__main__":
        unittest.main()
