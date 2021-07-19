"""Test N2 GUI with multiple models using Playwright."""
import unittest
import sys

from n2_gui_test import n2_gui_test_case

# Playwright requires Python 3.7 or higher
if sys.version_info > (3, 6):
    if __name__ == "__main__":
        unittest.main()
