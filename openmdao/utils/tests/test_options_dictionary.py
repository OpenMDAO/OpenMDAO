from openmdao.api import OptionsDictionary
import unittest
import warnings
from six import PY3, assertRegex


class TestOptionsDict(unittest.TestCase):

    def setUp(self):
        self.options = OptionsDictionary()

    def test_reprs(self):
        self.options.declare('test', types=int, desc='Test integer value')
        self.options.declare('flag', default=False, types=bool)
        self.options.declare('long_desc', types=str,
                             desc='This description is ridiculously long and verbose, so much '
                                  'so that it takes up multiple lines in the options table.')

        self.assertEqual(self.options.__repr__(), self.options._dict)

        self.assertEqual(self.options.__str__(width=80), '\n'.join([
            "========= ============ ================= ================ ======================",
            "Option    Default      Acceptable Values Acceptable Types Description           ",
            "========= ============ ================= ================ ======================",
            "flag      False        N/A               ['bool']                               ",
            "long_desc **Required** N/A               ['str']          This description is ri",
            "                                                          diculously long and ve",
            "                                                          rbose, so much so that",
            "                                                           it takes up multiple ",
            "                                                          lines in the options t",
            "                                                          able.",
            "test      **Required** N/A               ['int']          Test integer value    ",
            "========= ============ ================= ================ ======================"
        ]))

    def test_type_checking(self):
        self.options.declare('test', types=int, desc='Test integer value')

        self.options['test'] = 1
        self.assertEqual(self.options['test'], 1)

        with self.assertRaises(TypeError) as context:
            self.options['test'] = ''

        class_or_type = 'class' if PY3 else 'type'
        expected_msg = "Option 'test' has the wrong type (<{} 'int'>)".format(class_or_type)
        self.assertEqual(expected_msg, str(context.exception))

        # make sure bools work
        self.options.declare('flag', default=False, types=bool)
        self.assertEqual(self.options['flag'], False)
        self.options['flag'] = True
        self.assertEqual(self.options['flag'], True)

    def test_allow_none(self):
        self.options.declare('test', types=int, allow_none=True, desc='Test integer value')
        self.options['test'] = None
        self.assertEqual(self.options['test'], None)

    def test_type_and_values(self):
        # Test with only type_
        self.options.declare('test1', types=int)
        self.options['test1'] = 1
        self.assertEqual(self.options['test1'], 1)

        # Test with only values
        self.options.declare('test2', values=['a', 'b'])
        self.options['test2'] = 'a'
        self.assertEqual(self.options['test2'], 'a')

        # Test with both type_ and values
        with self.assertRaises(Exception) as context:
            self.options.declare('test3', types=int, values=['a', 'b'])
        self.assertEqual(str(context.exception),
                         "'types' and 'values' were both specified for option 'test3'.")

    def test_isvalid(self):
        self.options.declare('even_test', types=int, is_valid=lambda x: x % 2 == 0)
        self.options['even_test'] = 2
        self.options['even_test'] = 4

        with self.assertRaises(ValueError) as context:
            self.options['even_test'] = 3

        expected_msg = "Function is_valid returns False for {}.".format('even_test')
        self.assertEqual(expected_msg, str(context.exception))

    def test_isvalid_deprecated_type(self):

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.options.declare('even_test', type_=int, is_valid=lambda x: x % 2 == 0)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[-1].message), "In declaration of option 'even_test' the '_type' arg is deprecated.  Use 'types' instead.")

        self.options['even_test'] = 2
        self.options['even_test'] = 4

        with self.assertRaises(ValueError) as context:
            self.options['even_test'] = 3

        expected_msg = "Function is_valid returns False for {}.".format('even_test')
        self.assertEqual(expected_msg, str(context.exception))

    def test_unnamed_args(self):
        with self.assertRaises(KeyError) as context:
            self.options['test'] = 1

        # KeyError ends up with an extra set of quotes.
        expected_msg = "\"Key 'test' cannot be set because it has not been declared.\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_contains(self):
        self.options.declare('test')

        contains = 'undeclared' in self.options
        self.assertTrue(not contains)

        contains = 'test' in self.options
        self.assertTrue(contains)

    def test_update(self):
        self.options.declare('test', default='Test value', types=object)

        obj = object()
        self.options.update({'test': obj})
        self.assertIs(self.options['test'], obj)

    def test_update_extra(self):
        with self.assertRaises(KeyError) as context:
            self.options.update({'test': 2})

        # KeyError ends up with an extra set of quotes.
        expected_msg = "\"Key 'test' cannot be set because it has not been declared.\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_get_missing(self):
        with self.assertRaises(KeyError) as context:
            self.options['missing']

        expected_msg = "\"Option 'missing' cannot be found\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_get_default(self):
        obj_def = object()
        obj_new = object()

        self.options.declare('test', default=obj_def, types=object)

        self.assertIs(self.options['test'], obj_def)

        self.options['test'] = obj_new
        self.assertIs(self.options['test'], obj_new)

    def test_values(self):
        obj1 = object()
        obj2 = object()
        self.options.declare('test', values=[obj1, obj2])

        self.options['test'] = obj1
        self.assertIs(self.options['test'], obj1)

        with self.assertRaises(ValueError) as context:
            self.options['test'] = object()

        expected_msg = ("Option 'test''s value is not one of \[<object object at 0x[0-9A-Fa-f]+>,"
                        " <object object at 0x[0-9A-Fa-f]+>\]")
        assertRegex(self, str(context.exception), expected_msg)

    def test_read_only(self):
        opt = OptionsDictionary(read_only=True)
        opt.declare('permanent', 3.0)

        with self.assertRaises(KeyError) as context:
            opt['permanent'] = 4.0

        expected_msg = ("Tried to set 'permanent' on a read-only OptionsDictionary")
        assertRegex(self, str(context.exception), expected_msg)

    def test_bounds(self):
        self.options.declare('x', default=1.0, lower=0.0, upper=2.0)

        with self.assertRaises(ValueError) as context:
            self.options['x'] = 3.0

        expected_msg = ("Value of 3.0 exceeds maximum of 2.0 for option 'x'")
        assertRegex(self, str(context.exception), expected_msg)

        with self.assertRaises(ValueError) as context:
            self.options['x'] = -3.0

        expected_msg = ("Value of -3.0 exceeds minimum of 0.0 for option 'x'")
        assertRegex(self, str(context.exception), expected_msg)

    def test_undeclare(self):
        # create an entry in the dict
        self.options.declare('test', types=int)
        self.options['test'] = 1

        # prove it's in the dict
        self.assertEqual(self.options['test'], 1)

        # remove entry from the dict
        self.options.undeclare("test")

        # prove it is no longer in the dict
        with self.assertRaises(KeyError) as context:
            self.options['test']

        expected_msg = "\"Option 'test' cannot be found\""
        self.assertEqual(expected_msg, str(context.exception))


if __name__ == "__main__":
    unittest.main()
