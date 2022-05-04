"""Test N2 GUI with multiple models using Playwright."""
import unittest
import os

os.system("playwright install")
from n2_gui_test import n2_gui_test_case
from gen_gui_test import gen_gui_test_case

if __name__ == "__main__":
    unittest.main()
