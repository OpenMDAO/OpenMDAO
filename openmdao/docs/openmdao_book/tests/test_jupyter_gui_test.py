import unittest
import pathlib


@unittest.skipUnless(pathlib.Path(__file__).parent.parent.joinpath("_build").exists(), "Cannot test without docs being built")
class JupyterBookGuiTestCases(unittest.TestCase):
    """
    Use a GUI testing tool to test functionality of the docs.
    """
    def test_include_source_docs_option(self):
        a = 1/1
        pass