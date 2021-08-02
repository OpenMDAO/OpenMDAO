import unittest

from openmdao.utils.testing_utils import use_tempdirs, MissingImports


@use_tempdirs
class TestMissingImports(unittest.TestCase):

    def test_missing_imports(self):

        with self.assertRaises(ImportError) as e:
            with MissingImports('pyoptsparse'):
                import pyoptsparse

        msg = "No module named pyoptsparse due to missing import pyoptsparse."

        self.assertEqual(msg, str(e.exception))

    def test_missing_imports_docs(self):

        with self.assertRaises(ImportError) as e:
            with MissingImports('IPython'):
                import IPython

        msg = "No module named IPython due to missing import IPython."

        self.assertEqual(msg, str(e.exception))

        with self.assertRaises(ImportError) as e:
            with MissingImports('tabulate'):
                import tabulate

        msg = "No module named tabulate due to missing import tabulate."

        self.assertEqual(msg, str(e.exception))

        with self.assertRaises(ImportError) as e:
            with MissingImports('matplotlib'):
                import matplotlib

        msg = "No module named matplotlib due to missing import matplotlib."

        self.assertEqual(msg, str(e.exception))

        with self.assertRaises(ImportError) as e:
            with MissingImports('nbconvert'):
                import nbconvert

        msg = "No module named nbconvert due to missing import nbconvert."

        self.assertEqual(msg, str(e.exception))

        with self.assertRaises(ImportError) as e:
            with MissingImports('numpydoc'):
                import numpydoc

        msg = "No module named numpydoc due to missing import numpydoc."

        self.assertEqual(msg, str(e.exception))

    def test_missing_imports_notebooks(self):

        with self.assertRaises(ImportError) as e:
            with MissingImports('notebook'):
                import notebook

        msg = "No module named notebook due to missing import notebook."

        self.assertEqual(msg, str(e.exception))

    def test_missing_imports_visualization(self):
        with self.assertRaises(ImportError) as e:
            with MissingImports('bokeh'):
                import bokeh

        msg = "No module named bokeh due to missing import bokeh."

        self.assertEqual(msg, str(e.exception))

    def test_missing_imports_testing(self):
        with self.assertRaises(ImportError) as e:
            with MissingImports('parameterized'):
                import parameterized

        msg = "No module named parameterized due to missing import parameterized."

        self.assertEqual(msg, str(e.exception))

        with self.assertRaises(ImportError) as e:
            with MissingImports('pycodestyle'):
                import pycodestyle

        msg = "No module named pycodestyle due to missing import pycodestyle."

        self.assertEqual(msg, str(e.exception))

        with self.assertRaises(ImportError) as e:
            with MissingImports('testflo'):
                import testflo

        msg = "No module named testflo due to missing import testflo."

        self.assertEqual(msg, str(e.exception))
