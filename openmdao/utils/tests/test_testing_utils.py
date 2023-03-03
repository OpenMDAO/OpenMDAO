import unittest

from openmdao.utils.testing_utils import use_tempdirs, MissingImports


@use_tempdirs
class TestMissingImports(unittest.TestCase):

    def test_missing_imports_cm(self):
        with self.assertRaises(ImportError) as e:
            with MissingImports('pyoptsparse'):
                import pyoptsparse

        msg = "No module named pyoptsparse due to missing import pyoptsparse."

        self.assertEqual(msg, str(e.exception))

    def test_missing_imports(self):

        with MissingImports('pyoptsparse'):
            import openmdao.api as om

    def test_missing_imports_docs(self):

        with MissingImports('IPython'):
            import openmdao.api as om

        with MissingImports('matplotlib'):
            import openmdao.api as om

        with MissingImports('numpydoc'):
            import openmdao.api as om

    def test_missing_imports_notebooks(self):

        with MissingImports('notebook'):
            import openmdao.api as om

    def test_missing_imports_visualization(self):
        with MissingImports('bokeh'):
            import openmdao.api as om

    def test_missing_imports_testing(self):
        with MissingImports('parameterized'):
            import openmdao.api as om

        with MissingImports('pycodestyle'):
            import openmdao.api as om

        with MissingImports('testflo'):
            import openmdao.api as om
