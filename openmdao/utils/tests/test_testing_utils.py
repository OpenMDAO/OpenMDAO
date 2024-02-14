import unittest
import re

from openmdao.utils.testing_utils import use_tempdirs, MissingImports, snum_equal, \
    rel_num_diff, num_rgx, snum_iter


@use_tempdirs
@unittest.skip("Turns out messing with builtins.__import__ really is a bad idea.")
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


class TestPrimitives(unittest.TestCase):
    def test_num_regex(self):
        numstrs = [
            "0",
            "-3",
            "-3.",
            "-1e6",
            "1e6",
            "-1.e5",
            "1.e5",
            "3",
            "3.",
            "3.14",
            "-3.14",
            "3e-2",
            "3e+2",
            "3.e-2",
            "-3e-2",
            "-3.e-2",
            "3.14e-2",
            "-3.14e-2",
            "3.14e+2",
        ]

        # Check each string with the regex pattern
        for s in numstrs:
            m =  re.match(num_rgx, s)
            self.assertTrue(m is not None, f"{s} should be parsed as a number")
            self.assertEqual(m.start(), 0, f"{s} should be parsed as a whole number but only used {m.group()}")
            self.assertEqual(m.end(), len(s), f"{s} should be parsed as a whole number but only used {m.group()}")

    def test_num_regex_mixed(self):
        mixedstrs = [
            ("a5b", None),
            ("5a", "5"),
            ("_5", None),
            ("5_", "5"),
            ("34blue", "34"),
            ("red5", None),
            ("[3, 4, 5]", None),
            ("[4]", None),
            ("3:5", "3"),
            ("3..6", "3."),
        ]

        for s, expected in mixedstrs:
            m =  re.match(num_rgx, s)
            val = m if m is None else m.group()
            self.assertEqual(val, expected, f"first number should be {expected} but got {val}")


class TestSNumIter(unittest.TestCase):
    def test_snum_iter(self):
        # Test with a string containing both numbers and non-numbers
        s = "abc123def456"
        result = list(snum_iter(s))
        self.assertEqual(result, [('abc', False), (123.0, True), ('def', False), (456.0, True)])

        # Test with a string containing both numbers and non-numbers, ending in non-number
        s = "abc123def456xyz"
        result = list(snum_iter(s))
        self.assertEqual(result, [('abc', False), (123.0, True), ('def', False), (456.0, True), ('xyz', False)])

        # Test with a string containing only numbers
        s = "123456"
        result = list(snum_iter(s))
        self.assertEqual(result, [(123456.0, True)])

        # Test with a string containing only non-numbers
        s = "abcdef"
        result = list(snum_iter(s))
        self.assertEqual(result, [('abcdef', False)])

        # Test with a string containing numbers with decimal points
        s = "abc123.456def789.012"
        result = list(snum_iter(s))
        self.assertEqual(result, [('abc', False), (123.456, True), ('def', False), (789.012, True)])

        # Test with an empty string
        s = ""
        result = list(snum_iter(s))
        self.assertEqual(result, [])

        # Test with a string containing negative numbers
        s = "abc-123def-456"
        result = list(snum_iter(s))
        self.assertEqual(result, [('abc', False), (-123.0, True), ('def', False), (-456.0, True)])

        # Test with a string containing numbers in scientific notation
        s = "abc1.23e-4def5.67e+8"
        result = list(snum_iter(s))
        self.assertEqual(result, [('abc', False), (1.23e-4, True), ('def', False), (5.67e+8, True)])


class TestRelNumDiff(unittest.TestCase):
    def test_rel_num_diff_zero(self):
        # Test with both numbers being zero
        self.assertEqual(rel_num_diff(0.0, 0.0), 0.0)

    def test_rel_num_diff_one_zero(self):
        # Test with one number being zero and the other being non-zero
        self.assertEqual(rel_num_diff(0.0, 5.0), 1.0)
        self.assertEqual(rel_num_diff(5.0, 0.0), 1.0)

    def test_rel_num_diff_positive(self):
        # Test with both numbers being positive
        self.assertAlmostEqual(rel_num_diff(5.0, 10.0), 1.0)
        self.assertAlmostEqual(rel_num_diff(10.0, 5.0), 0.5)

    def test_rel_num_diff_negative(self):
        # Test with both numbers being negative
        self.assertAlmostEqual(rel_num_diff(-5.0, -10.0), 1.0)
        self.assertAlmostEqual(rel_num_diff(-10.0, -5.0), 0.5)

    def test_rel_num_diff_mixed(self):
        # Test with one number being positive and the other being negative
        self.assertEqual(rel_num_diff(5.0, -5.0), 2.0)
        self.assertEqual(rel_num_diff(-5.0, 5.0), 2.0)


class TestSNumEqual(unittest.TestCase):

    def test_snum_equal_no_numbers(self):
        # Test with strings that do not contain numbers
        self.assertTrue(snum_equal("abc", "abc"))
        self.assertFalse(snum_equal("abc", "def"))

    def test_snum_equal_with_numbers(self):
        # Test with strings that contain numbers
        self.assertTrue(snum_equal("abc123", "abc123"))
        self.assertFalse(snum_equal("abc123", "abc456"))

    def test_snum_equal_with_numbers_within_tolerance(self):
        # Test with strings that contain numbers within the tolerance
        self.assertTrue(snum_equal("abc123.000001", "abc123.000002", atol=1e-6))
        self.assertTrue(snum_equal("abc123.000001", "abc123.000002", rtol=1e-6))

    def test_snum_equal_with_numbers_outside_tolerance(self):
        # Test with strings that contain numbers outside the tolerance
        self.assertFalse(snum_equal("abc123.0001", "abc123.0002", atol=1e-6))
        self.assertFalse(snum_equal("abc123.0001", "abc123.0002", rtol=1e-6))

    def test_snum_equal_with_multiple_numbers(self):
        # Test with strings that contain multiple numbers
        self.assertTrue(snum_equal("abc123def456", "abc123def456"))
        self.assertFalse(snum_equal("abc123def456", "abc123def789"))

    def test_snum_equal_with_multiple_numbers_within_tolerance(self):
        # Test with strings that contain multiple numbers within the tolerance
        self.assertTrue(snum_equal("abc123.000001def456.000001", "abc123.000002def456.000002", atol=1e-6))
        self.assertTrue(snum_equal("abc123.000001def456.000001", "abc123.000002def456.000002", rtol=1e-6))

    def test_snum_equal_with_multiple_numbers_outside_tolerance(self):
        # Test with strings that contain multiple numbers outside the tolerance
        self.assertFalse(snum_equal("abc123.0001def456.0001", "abc123.0002def456.0002", atol=1e-6))
        self.assertFalse(snum_equal("abc123.0001def456.0001", "abc123.0002def456.0002", rtol=1e-6))

