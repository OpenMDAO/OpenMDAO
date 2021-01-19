from openmdao.api import OptionsDictionary

import unittest

from openmdao.utils.assert_utils import assert_warning, assert_no_warning

from openmdao.core.explicitcomponent import ExplicitComponent


def check_even(name, value):
    if value % 2 != 0:
        raise ValueError("Option '%s' with value %s is not an even number." % (name, value))


class TestOptionsDict(unittest.TestCase):

    def setUp(self):
        self.dict = OptionsDictionary()

    def test_reprs(self):
        class MyComp(ExplicitComponent):
            pass

        my_comp = MyComp()

        self.dict.declare('test', values=['a', 'b'], desc='Test integer value')
        self.dict.declare('flag', default=False, types=bool)
        self.dict.declare('comp', default=my_comp, types=ExplicitComponent)
        self.dict.declare('long_desc', types=str,
                          desc='This description is long and verbose, so it '
                               'takes up multiple lines in the options table.')

        self.assertEqual(repr(self.dict), repr(self.dict._dict))

        self.assertEqual(self.dict.__str__(width=83), '\n'.join([
            "========= ============ ================= ===================== ====================",
            "Option    Default      Acceptable Values Acceptable Types      Description         ",
            "========= ============ ================= ===================== ====================",
            "comp      MyComp       N/A               ['ExplicitComponent']                     ",
            "flag      False        [True, False]     ['bool']                                  ",
            "long_desc **Required** N/A               ['str']               This description is ",
            "                                                               long and verbose, so",
            "                                                                it takes up multipl",
            "                                                               e lines in the optio",
            "                                                               ns table.",
            "test      **Required** ['a', 'b']        N/A                   Test integer value  ",
            "========= ============ ================= ===================== ====================",
        ]))

        # if the table can't be represented in specified width, then we get the full width version
        self.assertEqual(self.dict.__str__(width=40), '\n'.join([
            "========= ============ ================= ===================== ====================="
            "==================================================================== ",
            "Option    Default      Acceptable Values Acceptable Types      Description          "
            "                                                                     ",
            "========= ============ ================= ===================== ====================="
            "==================================================================== ",
            "comp      MyComp       N/A               ['ExplicitComponent']                      "
            "                                                                     ",
            "flag      False        [True, False]     ['bool']                                   "
            "                                                                     ",
            "long_desc **Required** N/A               ['str']               This description is l"
            "ong and verbose, so it takes up multiple lines in the options table. ",
            "test      **Required** ['a', 'b']        N/A                   Test integer value   "
            "                                                                     ",
            "========= ============ ================= ===================== ====================="
            "==================================================================== ",
        ]))

    def test_type_checking(self):
        self.dict.declare('test', types=int, desc='Test integer value')

        self.dict['test'] = 1
        self.assertEqual(self.dict['test'], 1)

        with self.assertRaises(TypeError) as context:
            self.dict['test'] = ''

        expected_msg = "Value ('') of option 'test' has type 'str', " \
                       "but type 'int' was expected."
        self.assertEqual(expected_msg, str(context.exception))

        # multiple types are allowed
        self.dict.declare('test_multi', types=(int, float), desc='Test multiple types')

        self.dict['test_multi'] = 1
        self.assertEqual(self.dict['test_multi'], 1)
        self.assertEqual(type(self.dict['test_multi']), int)

        self.dict['test_multi'] = 1.0
        self.assertEqual(self.dict['test_multi'], 1.0)
        self.assertEqual(type(self.dict['test_multi']), float)

        with self.assertRaises(TypeError) as context:
            self.dict['test_multi'] = ''

        expected_msg = "Value ('') of option 'test_multi' has type 'str', " \
                       "but one of types ('int', 'float') was expected."
        self.assertEqual(expected_msg, str(context.exception))

        # make sure bools work and allowed values are populated
        self.dict.declare('flag', default=False, types=bool)
        self.assertEqual(self.dict['flag'], False)
        self.dict['flag'] = True
        self.assertEqual(self.dict['flag'], True)

        meta = self.dict._dict['flag']
        self.assertEqual(meta['values'], (True, False))

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

    def test_check_valid_template(self):
        # test the template 'check_valid' function
        from openmdao.utils.options_dictionary import check_valid
        self.dict.declare('test', check_valid=check_valid)

        with self.assertRaises(ValueError) as context:
            self.dict['test'] = 1

        expected_msg = "Option 'test' with value 1 is not valid."
        self.assertEqual(expected_msg, str(context.exception))

    def test_isvalid(self):
        self.dict.declare('even_test', types=int, check_valid=check_even)
        self.dict['even_test'] = 2
        self.dict['even_test'] = 4

        with self.assertRaises(ValueError) as context:
            self.dict['even_test'] = 3

        expected_msg = "Option 'even_test' with value 3 is not an even number."
        self.assertEqual(expected_msg, str(context.exception))

    def test_unnamed_args(self):
        with self.assertRaises(KeyError) as context:
            self.dict['test'] = 1

        # KeyError ends up with an extra set of quotes.
        expected_msg = "\"Option 'test' cannot be set because it has not been declared.\""
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
        expected_msg = "\"Option 'test' cannot be set because it has not been declared.\""
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

        expected_msg = ("Value \(<object object at 0x[0-9A-Fa-f]+>\) of option 'test' is not one of \[<object object at 0x[0-9A-Fa-f]+>,"
                        " <object object at 0x[0-9A-Fa-f]+>\].")
        self.assertRegex(str(context.exception), expected_msg)

    def test_read_only(self):
        opt = OptionsDictionary(read_only=True)
        opt.declare('permanent', 3.0)

        with self.assertRaises(KeyError) as context:
            opt['permanent'] = 4.0

        expected_msg = ("Tried to set read-only option 'permanent'.")
        self.assertRegex(str(context.exception), expected_msg)

    def test_bounds(self):
        self.dict.declare('x', default=1.0, lower=0.0, upper=2.0)

        with self.assertRaises(ValueError) as context:
            self.dict['x'] = 3.0

        expected_msg = "Value (3.0) of option 'x' exceeds maximum allowed value of 2.0."
        self.assertEqual(str(context.exception), expected_msg)

        with self.assertRaises(ValueError) as context:
            self.dict['x'] = -3.0

        expected_msg = "Value (-3.0) of option 'x' is less than minimum allowed value of 0.0."
        self.assertEqual(str(context.exception), expected_msg)

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

    def test_deprecated_option(self):
        msg = 'Option "test1" is deprecated.'
        self.dict.declare('test1', deprecation=msg)

        # test double set
        with assert_warning(DeprecationWarning, msg):
            self.dict['test1'] = None
        # Should only generate warning first time
        with assert_no_warning(DeprecationWarning, msg):
            self.dict['test1'] = None

        # Also test set and then get
        msg = 'Option "test2" is deprecated.'
        self.dict.declare('test2', deprecation=msg)

        with assert_warning(DeprecationWarning, msg):
            self.dict['test2'] = None
        # Should only generate warning first time
        with assert_no_warning(DeprecationWarning, msg):
            option = self.dict['test2']


if __name__ == "__main__":
    unittest.main()
