"""Test N2 GUI with multiple models using Pyppeteer."""
import sys
import unittest

# Make sure Python version is at least 3.6, otherwise do nothing.
if sys.version_info > (3, 5):
    from n2_gui_test import n2_gui_test_case

    if __name__ == "__main__":
        unittest.main()
