"""Test N2 GUI with multiple models using Playwright."""
import unittest
import os

from n2_gui_test import n2_gui_test_case

if __name__ == "__main__":
    os.system("playwright install")
    unittest.main()
