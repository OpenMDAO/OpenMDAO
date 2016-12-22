from openmdao.utils.generalized_dict import GeneralizedDictionary, OptionsDictionary
import unittest

class TestOptionsDict(unittest.TestCase):

    def setUp(self):
        self.dict = OptionsDictionary()

    def test_in_dict(self):
        expected_msg = 'Initial dictionaries cannot be used with OptionsDictionary declare options and use update.'
        with self.assertRaises(ValueError) as context:
            tmp = OptionsDictionary({'a':1})

        self.assertEqual(expected_msg, str(context.exception))

    def test_type_checking(self):
        self.dict.declare('test', int, 'Test integer value')

        self.dict['test'] = 1
        self.assertEqual(self.dict['test'], 1)

        with self.assertRaises(ValueError) as context:
            self.dict['test'] = ''

        expected_msg = "Entry 'test' has the wrong type (<class 'int'>)"
        self.assertEqual(expected_msg, str(context.exception))
