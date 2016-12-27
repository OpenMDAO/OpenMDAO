from openmdao.utils.generalized_dict import GeneralizedDictionary, OptionsDictionary
import unittest
from six import PY3, assertRegex


class TestOptionsDict(unittest.TestCase):

    def setUp(self):
        self.dict = OptionsDictionary()

    def test_in_dict(self):
        expected_msg = ('Initial dictionaries cannot be used with OptionsDictionary. '
                        'Declare options and use update.')
        with self.assertRaises(ValueError) as context:
            OptionsDictionary({'a': 1})

        self.assertEqual(expected_msg, str(context.exception))

    def test_type_checking(self):
        self.dict.declare('test', int, 'Test integer value')

        self.dict['test'] = 1
        self.assertEqual(self.dict['test'], 1)

        with self.assertRaises(ValueError) as context:
            self.dict['test'] = ''

        class_or_type = 'class' if PY3 else 'type'
        expected_msg = "Entry 'test' has the wrong type (<{} 'int'>)".format(class_or_type)
        self.assertEqual(expected_msg, str(context.exception))

    def test_unnamed_args(self):
        with self.assertRaises(KeyError) as context:
            self.dict['test'] = 1

        # KeyError ends up with an extra set of quotes.
        expected_msg = "\"Entry 'test' is not declared\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_update(self):
        self.dict.declare('test', object, 'Test value')

        obj = object()
        self.dict.update({'test': obj})
        self.assertIs(self.dict['test'], obj)

    def test_update_extra(self):
        with self.assertRaises(KeyError) as context:
            self.dict.update({'test': 2})

        # KeyError ends up with an extra set of quotes.
        expected_msg = "\"Entry '{}' is not declared\"".format('test')
        self.assertEqual(expected_msg, str(context.exception))

    def test_get_missing(self):
        with self.assertRaises(KeyError) as context:
            self.dict['missing']

        expected_msg = "\"Entry 'missing' cannot be found\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_get_default(self):
        obj_def = object()
        obj_new = object()

        self.dict.declare('test', object, value=obj_def)

        self.assertIs(self.dict['test'], obj_def)

        self.dict['test'] = obj_new
        self.assertIs(self.dict['test'], obj_new)

    def test_get_missing_default(self):
        self.dict.declare('test', object)

        with self.assertRaises(KeyError) as context:
            self.dict['test']

        expected_msg = "\"Entry 'test' has no default\""
        self.assertEqual(expected_msg, str(context.exception))

    def test_assemble_global(self):
        parent_dict = OptionsDictionary()
        parent_dict.declare('test', object)
        parent_dict.declare('test2', object)

        obj = object()
        parent_dict['test'] = obj
        parent_dict['test2'] = obj

        with self.assertRaises(KeyError) as context:
            self.dict._assemble_global_dict(parent_dict)

        expected_msg = "\"Entry 'test(2)?' is not declared\""
        assertRegex(self, str(context.exception), expected_msg)

        self.dict.declare('test', object)
        self.dict.declare('test2', object)
        self.dict._assemble_global_dict(parent_dict)

        self.assertIs(self.dict._global_dict['test'], obj)
        obj2 = object()
        self.dict['test2'] = obj2
        self.dict._assemble_global_dict(parent_dict)

        self.assertIs(self.dict._global_dict['test2'], obj2)

    def test_values(self):
        obj1 = object()
        obj2 = object()
        self.dict.declare('test', object, values=[obj1, obj2])

        self.dict['test'] = obj1
        self.assertIs(self.dict['test'], obj1)

        with self.assertRaises(ValueError) as context:
            self.dict['test'] = object()

        expected_msg = ("Entry 'test''s value is not one of \[<object object at 0x[0-9A-Fa-f]+>,"
                        " <object object at 0x[0-9A-Fa-f]+>\]")
        assertRegex(self, str(context.exception), expected_msg)


class TestGeneralizedDict(TestOptionsDict):
    def setUp(self):
        self.dict = GeneralizedDictionary()

    def test_in_dict(self):
        pass

    def test_unnamed_args(self):
        obj = object()
        self.dict['test'] = obj
        self.assertIs(self.dict['test'], obj)

    def test_update_extra(self):
        obj = object()
        self.dict.update({'test': obj})
        self.assertIs(self.dict['test'], obj)

    def test_assemble_global(self):
        parent_dict = OptionsDictionary()
        parent_dict.declare('test', object)
        parent_dict.declare('test2', object)

        obj = object()
        parent_dict['test'] = obj
        parent_dict['test2'] = obj
        self.dict._assemble_global_dict(parent_dict)

        self.assertIs(self.dict._global_dict['test'], obj)
        obj2 = object()
        self.dict['test2'] = obj2
        self.dict._assemble_global_dict(parent_dict)

        self.assertIs(self.dict._global_dict['test2'], obj2)