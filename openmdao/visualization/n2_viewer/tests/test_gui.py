"""Test N2 GUI with multiple models using Playwright."""
import sys
import unittest
import os
import urllib.request

# Make sure Python version is at least 3.6, otherwise do nothing.
if sys.version_info > (3, 5):
    from pyppeteer.chromium_downloader import check_chromium, download_chromium
    from appdirs import AppDirs
    import shutil

    from n2_gui_test import n2_gui_test_case

    if __name__ == "__main__":
        unittest.main()
