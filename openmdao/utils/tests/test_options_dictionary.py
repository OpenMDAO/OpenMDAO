from openmdao.api import OptionsDictionary
import unittest
import warnings
from six import PY3, assertRegex

from openmdao.core.explicitcomponent import ExplicitComponent


class TestOptionsDict(unittest.TestCase):

    def setUp(self):
        self.dict = OptionsDictionary()

    def test_reprs(self):
        my_comp = ExplicitComponent()

        self.dict.declare('test', values=['a', 'b'], desc='Test integer value')
        self.dict.declare('flag', default=False, types=bool)
        self.dict.declare('comp', default=my_comp, types=ExplicitComponent)
        self.dict.declare('long_desc', types=str,
                          desc='This description is long and verbose, so it '
                               'takes up multiple lines in the options table.')

        self.assertEqual(self.dict.__repr__(), self.dict._dict)

        self.assertEqual(self.dict.__str__(width=84), '\n'.join([
            "========= ================= ================= ===================== ================",
            "Option    Default           Acceptable Values Acceptable Types      Description     ",
            "========= ================= ================= ===================== ================",
            "comp      ExplicitComponent N/A               ['ExplicitComponent']                 ",
            "flag      False             N/A               ['bool']                              ",
            "long_desc **Required**      N/A               ['str']               This description",
            "                                                                     is long and ver",
            "                                                                    bose, so it take",
            "                                                                    s up multiple li",
            "                                                                    nes in the optio",
            "                                                                    ns table.",
            "test      **Required**      ['a', 'b']        N/A                   Test integer val",
            "                                                                    ue",
            "========= ================= ================= ===================== ================",
        ]))

    def test_type_checking(self):
        self.dict.declare('test', types=int, desc='Test integer value')

        self.dict['test'] = 1
        self.assertEqual(self.dict['test'], 1)

        with self.assertRaises(TypeError) as context:
            self.dict['test'] = ''

        class_or_type = 'class' if PY3 else 'type'
        expected_msg = "Option 'test' has the wrong type (<{} 'int'>)".format(class_or_type)
        self.assertEqual(expected_msg, str(context.exception))

        # make sure bools work
        self.dict.declare('flag', default=False, types=bool)
        self.assertEqual(self.dict['flag'], False)
        self.dict['flag'] = True
        self.assertEqual(self.dict['flag'], True)

    def test_allow_none(self):
        self.dict.declare('test', types=int, allow_none=True, desc='Test integer value')
        self.dict['test'] = None
        self.assertEqual(self.dict['test'], None)

    def test_type_and_values(self):
        # Test with only type_
        self.dict.declare('test1', types=int)
        self.dict['test1'] = 1
        self.assertEqual(self.dict['test1'], 1)

        # Test with only values
        self.dict.declare('test2', values=['a', 'b'])
        self.dict['test2'] = 'a'
        self.assertEqual(self.dict['test2'], 'a')

        # Test with both type_ and values
        with self.assertRaises(Exception) as context:
            self.dict.declare('test3', types=int, values=['a', 'b'])
        self.assertEqual(str(context.exception),
                         "'types' and 'values' were both specified for option 'test3'.")

    def test_isvalid(self):
        self.dict.declare('even_test', types=int, is_valid=lambda x: x % 2 == 0)
        self.dict['even_test'] = 2
        self.dict['even_test'] = 4

        with self.assertRaises(ValueError) as context:
            self.dict['even_test'] = 3

        expected_msg = "Function is_valid returns False for {}.".format('even_test')
        self.assertEqual(expected_msg, str(context.exception))

    def test_isvalid_deprecated_type(self):

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.dict.declare('even_test', type_=int, is_valid=lambda x: x % 2 == 0)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[-1].message), "In declaration of option 'even_test' the '_type' arg is deprecated.  Use 'types' instead.")

        self.dict['even_test'] = 2
        self.dict['even_test'] = 4

        with self.assertRaises(ValueError) as context:
            self.dict['even_test'] = 3

        expected_msg = "Function is_valid returns False for {}.".format('even_test')
        self.assertEqual(expected_msg, str(context.exception))

    def test_unnamed_args(self):
        with self.assertRaises(KeyError) as context:
            self.dict['test'] = 1

        # KeyError ends up with an extra set of quotes.
        expected_msg = "\"Key 'test' cannot be set because it has not been declared.\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_contains(self):
        self.dict.declare('test')

        contains = 'undeclared' in self.dict
        self.assertTrue(not contains)

        contains = 'test' in self.dict
        self.assertTrue(contains)

    def test_update(self):
        self.dict.declare('test', default='Test value', types=object)

        obj = object()
        self.dict.update({'test': obj})
        self.assertIs(self.dict['test'], obj)

    def test_update_extra(self):
        with self.assertRaises(KeyError) as context:
            self.dict.update({'test': 2})

        # KeyError ends up with an extra set of quotes.
        expected_msg = "\"Key 'test' cannot be set because it has not been declared.\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_get_missing(self):
        with self.assertRaises(KeyError) as context:
            self.dict['missing']

        expected_msg = "\"Option 'missing' cannot be found\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_get_default(self):
        obj_def = object()
        obj_new = object()

        self.dict.declare('test', default=obj_def, types=object)

        self.assertIs(self.dict['test'], obj_def)

        self.dict['test'] = obj_new
        self.assertIs(self.dict['test'], obj_new)

    def test_values(self):
        obj1 = object()
        obj2 = object()
        self.dict.declare('test', values=[obj1, obj2])

        self.dict['test'] = obj1
        self.assertIs(self.dict['test'], obj1)

        with self.assertRaises(ValueError) as context:
            self.dict['test'] = object()

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
        self.dict.declare('x', default=1.0, lower=0.0, upper=2.0)

        with self.assertRaises(ValueError) as context:
            self.dict['x'] = 3.0

        expected_msg = ("Value of 3.0 exceeds maximum of 2.0 for option 'x'")
        assertRegex(self, str(context.exception), expected_msg)

        with self.assertRaises(ValueError) as context:
            self.dict['x'] = -3.0

        expected_msg = ("Value of -3.0 exceeds minimum of 0.0 for option 'x'")
        assertRegex(self, str(context.exception), expected_msg)

    def test_undeclare(self):
        # create an entry in the dict
        self.dict.declare('test', types=int)
        self.dict['test'] = 1

        # prove it's in the dict
        self.assertEqual(self.dict['test'], 1)

        # remove entry from the dict
        self.dict.undeclare("test")

        # prove it is no longer in the dict
        with self.assertRaises(KeyError) as context:
            self.dict['test']

        expected_msg = "\"Option 'test' cannot be found\""
        self.assertEqual(expected_msg, str(context.exception))


if __name__ == "__main__":
    unittest.main()
