"""
Testing the file wrapping utilities.
"""

import os
import tempfile
import shutil

import unittest

from openmdao.utils.assert_utils import assert_near_equal, assert_equal_arrays

import numpy
from numpy import array, isnan, isinf

from openmdao.utils.file_wrap import InputFileGenerator, FileParser

DIRECTORY = os.path.dirname((os.path.abspath(__file__)))


class TestCase(unittest.TestCase):
    """ Test file wrapping functions. """

    def setUp(self):
        self.templatename = 'template.dat'
        self.filename = 'filename.dat'
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='omdao-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        if not os.environ.get('OPENMDAO_KEEPDIRS', False):
            try:
                shutil.rmtree(self.tempdir)
            except OSError:
                pass

    def test_templated_input(self):
        template = '\n'.join([
            "Junk",
            "Anchor",
            " A 1, 2 34, Test 1e65",
            " B 4 Stuff",
            "Anchor",
            " C 77 False Inf 333.444"
        ])

        outfile = open(self.templatename, 'w')
        outfile.write(template)
        outfile.close()

        gen = InputFileGenerator()
        gen.set_template_file(self.templatename)
        gen.set_generated_file(self.filename)
        gen.set_delimiters(', ')

        gen.mark_anchor('Anchor')
        gen.transfer_var('CC', 2, 0)
        gen.transfer_var(3.0, 1, 3)
        gen.reset_anchor()
        gen.mark_anchor('Anchor', 2)
        gen.transfer_var('NaN', 1, 4)
        gen.reset_anchor()
        gen.transfer_var('55', 3, 2)
        gen.mark_anchor('C 77')
        gen.transfer_var(1.3e-37, -3, 6)
        gen.clearline(-5)
        gen.mark_anchor('Anchor', -1)
        gen.transfer_var('8.7', 1, 5)

        gen.generate()

        infile = open(self.filename, 'r')
        result = infile.read()
        infile.close()

        answer = '\n'.join([
            "",
            "Anchor",
            " A 1, 3.0 34, Test 1.3e-37",
            " B 55 Stuff",
            "Anchor",
            " C 77 False NaN 8.7"
        ])

        self.assertEqual(answer, result)

        # Test some errors
        try:
            gen.mark_anchor('C 77', 3.14)
        except ValueError as err:
            msg = "The value for occurrence must be an integer"
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

        try:
            gen.mark_anchor('C 77', 0)
        except ValueError as err:
            msg = "0 is not valid for an anchor occurrence."
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

        try:
            gen.mark_anchor('ZZZ')
        except RuntimeError as err:
            msg = "Could not find pattern ZZZ in template file template.dat"
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected')

    def test_templated_input_same_anchors(self):
        template = '\n'.join([
            "CQUAD4 1 3.456",
            "CQUAD4 2 4.123",
            "CQUAD4 3 7.222",
            "CQUAD4 4"
        ])

        outfile = open(self.templatename, 'w')
        outfile.write(template)
        outfile.close()

        gen = InputFileGenerator()
        gen.set_template_file(self.templatename)
        gen.set_generated_file(self.filename)
        gen.set_delimiters(', ')

        gen.mark_anchor('CQUAD4')
        gen.transfer_var('x', 0, 2)
        gen.mark_anchor('CQUAD4')
        gen.transfer_var('y', 0, 3)
        gen.mark_anchor('CQUAD4', 2)
        gen.transfer_var('z', 0, 2)

        gen.generate()

        infile = open(self.filename, 'r')
        result = infile.read()
        infile.close()

        answer = '\n'.join([
            "CQUAD4 x 3.456",
            "CQUAD4 2 y",
            "CQUAD4 3 7.222",
            "CQUAD4 z"
        ])

        self.assertEqual(answer, result)

    def test_templated_input_arrays(self):
        template = '\n'.join([
            "Anchor",
            "0 0 0 0 0"
        ])

        outfile = open(self.templatename, 'w')
        outfile.write(template)
        outfile.close()

        gen = InputFileGenerator()
        gen.set_template_file(self.templatename)
        gen.set_generated_file(self.filename)

        gen.mark_anchor('Anchor')
        gen.transfer_array(array([1, 2, 3, 4.75, 5.0]), 1, 3, 5, sep=' ')

        gen.generate()

        infile = open(self.filename, 'r')
        result = infile.read()
        infile.close()

        answer = '\n'.join([
            "Anchor",
            "0 0 1.0 2.0 3.0 4.75 5.0"
        ])

        self.assertEqual(answer, result)

    def test_templated_input_2Darrays(self):
        template = '\n'.join([
            "Anchor",
            "0 0 0 0 0",
            "0 0 0 0 0"
        ])

        outfile = open(self.templatename, 'w')
        outfile.write(template)
        outfile.close()

        gen = InputFileGenerator()
        gen.set_template_file(self.templatename)
        gen.set_generated_file(self.filename)

        gen.mark_anchor('Anchor')
        var = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        gen.transfer_2Darray(var, 1, 2, 1, 5)

        gen.generate()

        infile = open(self.filename, 'r')
        result = infile.read()
        infile.close()

        answer = '\n'.join([
            "Anchor",
            "1 2 3 4 5",
            "6 7 8 9 10"
        ])

        self.assertEqual(answer, result)

    def test_output_parse(self):
        data = '\n'.join([
            "Junk",
            "Anchor",
            " A 1, 2 34, Test 1e65",
            " B 4 Stuff",
            "Anchor",
            " C 77 False NaN 333.444",
            " 1,2,3,4,5",
            " Inf 1.#QNAN -1.#IND"
        ])

        outfile = open(self.filename, 'w')
        outfile.write(data)
        outfile.close()

        gen = FileParser()
        gen.set_file(self.filename)
        gen.set_delimiters(' ')

        gen.mark_anchor('Anchor')
        val = gen.transfer_var(1, 1)
        self.assertEqual(val, 'A')
        gen.reset_anchor()
        val = gen.transfer_var(3, 2)
        self.assertEqual(val, 4)
        self.assertEqual(type(val), int)
        gen.mark_anchor('Anchor',2)
        val = gen.transfer_var(1, 4)
        self.assertEqual(isnan(val), True)
        val = gen.transfer_var(3, 1)
        self.assertEqual(isinf(val), True)
        val = gen.transfer_var(3, 2)
        self.assertEqual(isnan(val), True)
        val = gen.transfer_var(3, 3)
        self.assertEqual(isnan(val), True)
        val = gen.transfer_line(-1)
        self.assertEqual(val, ' B 4 Stuff')

        # Now, let's try column delimiters
        gen.set_delimiters('columns')
        gen.mark_anchor('Anchor',-1)
        val = gen.transfer_var(1, 8, 10)
        self.assertEqual(val, 'als')
        val = gen.transfer_var(1, 17)
        self.assertEqual(val, 333.444)

        # Test some errors
        try:
            gen.mark_anchor('C 77', 3.14)
        except ValueError as err:
            msg = "The value for occurrence must be an integer"
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

        try:
            gen.mark_anchor('C 77', 0)
        except ValueError as err:
            msg = "0 is not valid for an anchor occurrence."
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

        try:
            gen.mark_anchor('ZZZ')
        except RuntimeError as err:
            msg = "Could not find pattern ZZZ in output file filename.dat"
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected')

    def test_output_parse_same_anchors(self):
        data = '\n'.join([
            "CQUAD4 1 3.456",
            "CQUAD4 2 4.123",
            "CQUAD4 3 7.222",
            "CQUAD4 4"
        ])

        outfile = open(self.filename, 'w')
        outfile.write(data)
        outfile.close()

        gen = FileParser()
        gen.set_file(self.filename)
        gen.set_delimiters(' ')

        gen.mark_anchor('CQUAD4')
        val = gen.transfer_var(0, 3)
        self.assertEqual(val, 3.456)

        gen.mark_anchor('CQUAD4')
        val = gen.transfer_var(0, 3)
        self.assertEqual(val, 4.123)

        gen.mark_anchor('CQUAD4', 2)
        val = gen.transfer_var(0, 2)
        self.assertEqual(val, 4)

        gen.reset_anchor()

        gen.mark_anchor('CQUAD4', -1)
        val = gen.transfer_var(0, 2)
        self.assertEqual(val, 4)

        gen.mark_anchor('CQUAD4', -1)
        val = gen.transfer_var(0, 3)
        self.assertEqual(val, 7.222)

        gen.mark_anchor('CQUAD4', -2)
        val = gen.transfer_var(0, 3)
        self.assertEqual(val, 4.123)

    def test_output_parse_keyvar(self):
        data = '\n'.join([
            "Anchor",
            " Key1 1 2 3.7 Test 1e65",
            " Key1 3 4 3.2 ibg 0.0003",
            " Key1 5 6 6.7 Tst xxx"
        ])

        outfile = open(self.filename, 'w')
        outfile.write(data)
        outfile.close()

        gen = FileParser()
        gen.set_file(self.filename)
        gen.set_delimiters(' ')

        gen.mark_anchor('Anchor')
        val = gen.transfer_keyvar('Key1', 3)
        self.assertEqual(val, 3.7)
        val = gen.transfer_keyvar('Key1', 4, -2)
        self.assertEqual(val, 'ibg')
        val = gen.transfer_keyvar('Key1', 4, -2, -1)
        self.assertEqual(val, 'Test')

        try:
            gen.transfer_keyvar('Key1', 4, 0)
        except ValueError as err:
            msg = "The value for occurrence must be a nonzero integer"
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

        try:
            gen.transfer_keyvar('Key1', 4, -3.4)
        except ValueError as err:
            msg = "The value for occurrence must be a nonzero integer"
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

    def test_output_parse_array(self):
        data = '\n'.join([
            "Anchor",
            "10 20 30 40 50 60 70 80",
            "11 21 31 41 51 61 71 81",
            "Key a b c d e"
        ])

        outfile = open(self.filename, 'w')
        outfile.write(data)
        outfile.close()

        gen = FileParser()
        gen.set_file(self.filename)
        gen.set_delimiters(' ')

        gen.mark_anchor('Anchor')
        val = gen.transfer_array(1, 1, 1, 8)
        self.assertEqual(val[0], 10)
        self.assertEqual(val[7], 80)
        val = gen.transfer_array(1, 5, 2, 6)
        self.assertEqual(val[0], 50)
        self.assertEqual(val[9], 61)
        gen.mark_anchor('Key')
        val = gen.transfer_array(0, 2, 0, 6)
        self.assertEqual(val[4], 'e')
        val = gen.transfer_array(0, 2, fieldend=6)
        self.assertEqual(val[4], 'e')

        # Now, let's try column delimiters
        gen.reset_anchor()
        gen.mark_anchor('Anchor')
        gen.set_delimiters('columns')
        val = gen.transfer_array(1, 7, 1, 15)
        self.assertEqual(val[0], 30)
        self.assertEqual(val[2], 50)
        val = gen.transfer_array(1, 10, 2, 18)
        self.assertEqual(val[0], 40)
        self.assertEqual(val[5], 61)
        val = gen.transfer_array(3, 5, 3, 10)
        self.assertEqual(val[0], 'a b c')

        try:
            gen.transfer_array(1, 7, 1)
        except ValueError as err:
            msg = "fieldend is missing, currently required"
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

    def test_output_parse_2Darray(self):
        data = '''
        Anchor
            FREQ  DELTA  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5
             Hz
             50.   1.0   30.0  34.8  36.3  36.1  34.6  32.0  28.4  23.9  18.5  12.2   5.0  -3.1 -12.3 -22.5 -34.0 -47.2 -63.7
             63.   1.0   36.5  41.3  42.8  42.6  41.1  38.5  34.9  30.4  25.0  18.7  11.5   3.4  -5.8 -16.0 -27.5 -40.7 -57.2
             80.   1.0   42.8  47.6  49.1  48.9  47.4  44.8  41.2  36.7  31.3  25.0  17.8   9.7   0.5  -9.7 -21.2 -34.4 -50.9
            100.   1.0   48.4  53.1  54.7  54.5  53.0  50.4  46.8  42.3  36.9  30.6  23.3  15.2   6.1  -4.2 -15.7 -28.9 -45.4
            125.   1.0   53.6  58.3  59.9  59.6  58.1  55.5  52.0  47.5  42.0  35.7  28.5  20.4  11.2   1.0 -10.5 -23.7 -40.2
            160.   1.0   58.9  63.7  65.2  65.0  63.5  60.9  57.3  52.8  47.4  41.0  33.8  25.7  16.5   6.3  -5.2 -18.4 -34.9
            200.   1.0   63.4  68.1  69.6  69.4  67.9  65.3  61.7  57.2  51.8  45.5  38.3  30.1  21.0  10.7  -0.8 -14.0 -30.5
            250.   1.0   67.5  72.2  73.7  73.5  72.0  69.4  65.8  61.3  55.9  49.5  42.3  34.2  25.0  14.8   3.3 -10.0 -26.5
            315.   1.0   71.3  76.1  77.6  77.4  75.8  73.2  69.7  65.1  59.7  53.4  46.1  38.0  28.8  18.6   7.1  -6.2 -22.7
            400.   1.0   74.9  79.7  81.2  81.0  79.4  76.8  73.2  68.7  63.2  56.9  49.7  41.5  32.4  22.1  10.6  -2.7 -19.2
            500.   1.0   77.9  82.7  84.2  83.9  82.4  79.8  76.2  71.6  66.2  59.8  52.6  44.4  35.3  25.0  13.5   0.2 -16.3
            630.   1.0   80.7  85.4  86.9  86.6  85.1  82.4  78.8  74.3  68.8  62.5  55.2  47.0  37.9  27.6  16.1   2.8 -13.7
            800.   1.0   83.1  87.8  89.2  89.0  87.4  84.8  81.2  76.6  71.1  64.8  57.5  49.3  40.1  29.9  18.3   5.0 -11.5
           1000.   1.0   84.9  89.6  91.1  90.8  89.2  86.6  82.9  78.4  72.9  66.5  59.2  51.0  41.8  31.5  20.0   6.6  -9.9
           1250.   1.0   86.4  91.1  92.5  92.2  90.7  88.0  84.3  79.7  74.2  67.8  60.5  52.3  43.1  32.8  21.2   7.9  -8.7
           1600.   1.0   87.6  92.3  93.7  93.4  91.8  89.1  85.4  80.8  75.2  68.8  61.5  53.3  44.0  33.7  22.1   8.7  -7.9
           2000.   1.0   88.4  93.0  94.4  94.0  92.4  89.6  85.9  81.3  75.7  69.3  61.9  53.7  44.4  34.0  22.4   9.0  -7.6
           2500.   1.0   88.7  93.3  94.6  94.2  92.6  89.8  86.1  81.4  75.8  69.3  61.9  53.6  44.3  33.9  22.2   8.8  -7.9
           3150.   1.0   88.7  93.2  94.5  94.1  92.4  89.5  85.7  81.0  75.4  68.8  61.4  53.0  43.7  33.3  21.5   8.1  -8.6
           4000.   1.0   88.3  92.7  94.0  93.5  91.7  88.8  85.0  80.2  74.5  67.9  60.4  52.0  42.5  32.0  20.2   6.7 -10.0
           5000.   1.0   87.5  91.9  93.1  92.5  90.7  87.7  83.8  78.9  73.2  66.5  58.9  50.4  40.9  30.4  18.5   4.9 -11.9
           6300.   1.0   86.5  90.8  91.9  91.2  89.3  86.2  82.2  77.3  71.4  64.6  57.0  48.4  38.8  28.1  16.2   2.5 -14.5
           8000.   1.0   85.3  89.5  90.4  89.6  87.6  84.4  80.2  75.2  69.2  62.3  54.5  45.8  36.1  25.3  13.2  -0.6 -17.7
          10000.   1.0   84.2  88.2  89.0  88.1  85.9  82.5  78.3  73.0  66.9  59.9  51.9  43.1  33.2  22.3  10.1  -3.9 -21.1
        '''

        outfile = open(self.filename, 'w')
        outfile.write(data)
        outfile.close()

        gen = FileParser()
        gen.set_file(self.filename)

        # whitespace delim; with end field
        gen.set_delimiters(' \t')
        gen.mark_anchor('Anchor')
        val = gen.transfer_2Darray(3, 2, 26, 19)
        self.assertEqual(val[0, 1], 30.0)
        self.assertEqual(val[0, 17], -63.7)
        self.assertEqual(val[1, 17], -57.2)
        self.assertEqual(val[23, 17], -21.1)
        self.assertEqual(val.shape[0], 24)
        self.assertEqual(val.shape[1], 18)

        # whitespace delim; no end field
        val = gen.transfer_2Darray(3, 2, 26)
        self.assertEqual(val[0, 1], 30.0)
        self.assertEqual(val[23, 17], -21.1)
        self.assertEqual(val.shape[0], 24)
        self.assertEqual(val.shape[1], 18)

        # column delim; with end field
        gen.set_delimiters('columns')
        val = gen.transfer_2Darray(3, 19, 26, 125)
        self.assertEqual(val[0, 1], 30.0)
        self.assertEqual(val[0, 17], -63.7)
        self.assertEqual(val[1, 17], -57.2)
        self.assertEqual(val[23, 17], -21.1)
        self.assertEqual(val.shape[0], 24)
        self.assertEqual(val.shape[1], 18)

        # column delim; no end field
        val = gen.transfer_2Darray(3, 19, 26)
        self.assertEqual(val[0, 1], 30.0)
        self.assertEqual(val[0, 17], -63.7)
        self.assertEqual(val[1, 17], -57.2)
        self.assertEqual(val[23, 17], -21.1)
        self.assertEqual(val.shape[0], 24)
        self.assertEqual(val.shape[1], 18)

        # make sure single line works
        gen.set_delimiters(' \t')
        val = gen.transfer_2Darray(5, 3, 5, 5)
        self.assertEqual(val[0, 2], 49.1)

        # Small block read
        val = gen.transfer_2Darray(7, 3, 9, 6)
        self.assertEqual(val[0, 0], 53.6)
        self.assertEqual(val[2, 0], 63.4)

        # Error messages for bad values
        try:
            gen.transfer_2Darray(7, 3, 9, 1)
        except ValueError as err:
            msg = "fieldend must be greater than fieldstart"
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

        try:
            gen.transfer_2Darray(9, 2, 8, 4)
        except ValueError as err:
            msg = "rowend must be greater than rowstart"
            self.assertEqual(str(err), msg)
        else:
            self.fail('ValueError expected')

    def test_comment_char(self):
        # Check to see if the use of the comment
        #   characters works
        data = '\n'.join([
            "Junk",
            "CAnchor",
            " Z 11, 22 344, Test 1e65",
            " B 4 Stuff",
            "  $ Anchor",
            " Q 1, 2 34, Test 1e65",
            " B 4 Stuff",
            "Anchor",
            " A 1, 2 34, Test 1e65",
            " B 4 Stuff",
            "Anchor",
            " C 77 False NaN 333.444",
            " 1,2,3,4,5",
            " Inf 1.#QNAN -1.#IND"
        ])

        outfile = open(self.filename, 'w')
        outfile.write(data)
        outfile.close()

        # Test full line comments
        gen = FileParser(full_line_comment_char="C")
        gen.set_file(self.filename)
        gen.set_delimiters(' ')
        gen.mark_anchor('Anchor')
        val = gen.transfer_var(1, 1)
        self.assertEqual(val, 'A')

        # Test end of line comments also
        gen = FileParser(full_line_comment_char="C", end_of_line_comment_char="$")
        gen.set_file(self.filename)
        gen.set_delimiters(' ')
        gen.mark_anchor('Anchor')
        val = gen.transfer_var(1, 1)
        self.assertEqual(val, 'A')

    def test_more_delims(self):
        data = '\n'.join([
            "anchor,1.0,2.0",
            "abc=123.456",
            "c=1,2,Word,6",
            "d=C:/abc/def,a+b*c^2,(%#%),!true",
            "a^33 1.#QNAN^#$%^"
        ])

        outfile = open(self.filename, 'w')
        outfile.write(data)
        outfile.close()

        op = FileParser()
        op.set_file(self.filename)

        op.set_delimiters(' \t,=')

        op.mark_anchor('anchor')

        val = op.transfer_var(0, 1)
        self.assertEqual(val, 'anchor')
        val = op.transfer_var(0, 2)
        self.assertEqual(val, 1.0)
        val = op.transfer_var(1, 1)
        self.assertEqual(val, 'abc')
        val = op.transfer_var(1, 2)
        self.assertEqual(val, 123.456)
        val = op.transfer_var(2, 4)
        self.assertEqual(val, 'Word')
        val = op.transfer_var(2, 5)
        self.assertEqual(val, 6)
        val = op.transfer_var(3, 2)
        self.assertEqual(val, 'C:/abc/def')
        val = op.transfer_var(3, 3)
        self.assertEqual(val, 'a+b*c^2')
        val = op.transfer_var(3, 4)
        self.assertEqual(val, '(%#%)')
        val = op.transfer_var(3, 5)
        self.assertEqual(val, '!true')

        op.set_delimiters(' \t^')
        val = op.transfer_var(4, 1)
        self.assertEqual(val, 'a')
        val = op.transfer_var(4, 2)
        self.assertEqual(val, 33)
        val = op.transfer_var(4, 3)
        self.assertEqual(isnan(val), True)
        val = op.transfer_var(4, 4)
        self.assertEqual(val, '#$%')


class FileGenFeature(unittest.TestCase):

    # output data for each test
    output_data = {
        "": [
            "INPUT",
            "1 2 3",
            "INPUT",
            "10.1 20.2 30.3",
            "A B C"
        ],
        "test_transfer": [
            "INPUT",
            "1 7 3",
            "INPUT",
            "10.1 20.2 30.3",
            "A B C"
        ],
        "test_transfer_2": [
            "INPUT",
            "1 7 3",
            "INPUT",
            "10.1 20.2 3.141592653589793",
            "A B C"
        ],
        "test_transfer_minus2": [
            "INPUT",
            "99999 7 3",
            "INPUT",
            "10.1 20.2 3.141592653589793",
            "A B C"
        ],
        "test_transfer_array": [
            "INPUT",
            "123 456 789",
            "INPUT",
            "10.1 20.2 3.141592653589793",
            "A B C"
        ],
        "test_transfer_stretch": [
            "INPUT",
            "11 22 33 44 55 66",
            "INPUT",
            "10.1 20.2 3.141592653589793",
            "A B C"
        ]
    }

    # the name of the preceding test in the feature doc
    prev_test = {
        "test_transfer": "",
        "test_transfer_2": "test_transfer",
        "test_transfer_minus2": "test_transfer_2",
        "test_transfer_array": "test_transfer_minus2",
        "test_transfer_stretch": "test_transfer_array"
    }

    def setUp(self):
        from openmdao.utils.file_wrap import InputFileGenerator

        global parser  # global so we don't need `self.` in feature doc
        parser = InputFileGenerator()

        # the input data for each test is the output of the previous test
        prev_test = self.prev_test[self._testMethodName]
        parser._data = self.output_data[prev_test][:]

    def test_transfer(self):
        parser.mark_anchor("INPUT")
        parser.transfer_var(7, 1, 2)
        self.assertEqual(parser.generate(),
                         '\n'.join(self.output_data[self._testMethodName]))

    def test_transfer_2(self):
        parser.mark_anchor("INPUT", 2)

        my_var = 3.1415926535897932
        parser.transfer_var(my_var, 1, 3)

        self.assertEqual(parser.generate(),
                         '\n'.join(self.output_data[self._testMethodName]))

    def test_transfer_minus2(self):
        parser.reset_anchor()
        parser.mark_anchor("INPUT", -2)
        parser.transfer_var("99999", 1, 1)

        self.assertEqual(parser.generate(),
                         '\n'.join(self.output_data[self._testMethodName]))

    def test_transfer_array(self):
        from numpy import array

        array_val = array([123, 456, 789])

        parser.reset_anchor()
        parser.mark_anchor("INPUT")
        parser.transfer_array(array_val, 1, 1, 3)

        self.assertEqual(parser.generate(),
                         '\n'.join(self.output_data[self._testMethodName]))

    def test_transfer_stretch(self):
        from numpy import array

        array_val = array([11, 22, 33, 44, 55, 66])

        parser.reset_anchor()
        parser.mark_anchor("INPUT")
        parser.transfer_array(array_val, 1, 1, 3, sep=' ')

        self.assertEqual(parser.generate(),
                         '\n'.join(self.output_data[self._testMethodName]))


class FileParserFeature(unittest.TestCase):

    def setUp(self):
        import numpy
        from openmdao.utils.file_wrap import FileParser

        global parser  # global so we don't need `self.` in feature doc
        parser = FileParser()

        parser._data = [
            "LOAD CASE 1",
            "STRESS 1.3334e7 3.9342e7 NaN 2.654e5",
            "DISPLACEMENT 2.1 4.6 3.1 2.22234",
            "LOAD CASE 2",
            "STRESS 11 22 33 44 55 66",
            "DISPLACEMENT 1.0 2.0 3.0 4.0 5.0"
        ]

    def assert_equal_arrays(self, a1, a2):
        assert_equal_arrays(a1, a2)

    def test_parse_output(self):
        parser.mark_anchor("LOAD CASE")
        var = parser.transfer_var(1, 2)

        self.assertEqual((var, type(var)), (1.3334e+07, float))

    def test_parse_nan(self):
        parser.reset_anchor()
        parser.mark_anchor("LOAD CASE")
        var = parser.transfer_var(1, 4)

        from numpy import isnan, isinf
        self.assertEqual(isnan(var), True)

    def test_parse_string(self):
        parser.reset_anchor()
        parser.mark_anchor("LOAD CASE")
        var = parser.transfer_var(2, 1)

        self.assertEqual((var, type(var)), ("DISPLACEMENT", str))

    def test_parse_output_2(self):
        parser.reset_anchor()
        parser.mark_anchor("LOAD CASE", 2)
        var = parser.transfer_var(1, 2)

        self.assertEqual((var, type(var)), (11, int))

    def test_parse_output_minus2(self):
        parser.reset_anchor()
        parser.mark_anchor("LOAD CASE", -2)
        var = parser.transfer_var(1, 2)

        self.assertAlmostEqual(var, 1.3334e+07)

    def test_parse_keyvar(self):
        parser.reset_anchor()
        parser.mark_anchor("LOAD CASE 1")
        var = parser.transfer_keyvar("DISPLACEMENT", 1)

        self.assertEqual(var, 2.1)

    def test_parse_array(self):
        parser.reset_anchor()
        parser.mark_anchor("LOAD CASE")
        var = parser.transfer_array(2, 2, 2, 5)

        assert_near_equal(var, numpy.array([2.1, 4.6, 3.1, 2.22234]))

    def test_parse_array_multiline(self):
        parser.reset_anchor()
        parser.mark_anchor("LOAD CASE")
        var = parser.transfer_array(1, 3, 2, 4)

        self.assert_equal_arrays(var, numpy.array([
            '39342000.0', 'nan', '265400.0',
            'DISPLACEMENT', '2.1', '4.6', '3.1'
        ]))


class FileParser2dFeature(unittest.TestCase):

    def setUp(self):
        import numpy
        from openmdao.utils.file_wrap import FileParser

        global parser  # global so we don't need `self.` in feature doc
        parser = FileParser()

        # A way to "cheat" and do this without a file.
        parser._data = []
        parser._data.append('FREQ  DELTA  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5  -8.5')
        parser._data.append(' Hz')
        parser._data.append(' 50.   1.0   30.0  34.8  36.3  36.1  34.6  32.0  28.4  23.9  18.5  12.2')
        parser._data.append(' 63.   1.0   36.5  41.3  42.8  42.6  41.1  38.5  34.9  30.4  25.0  18.7')
        parser._data.append(' 80.   1.0   42.8  47.6  49.1  48.9  47.4  44.8  41.2  36.7  31.3  25.0')
        parser._data.append('100.   1.0   48.4  53.1  54.7  54.5  53.0  50.4  46.8  42.3  36.9  30.6')

    def assert_equal_arrays(self, a1, a2):
        assert_equal_arrays(a1, a2)

    def test_parse_array_2d(self):
        parser.reset_anchor()
        parser.mark_anchor("Hz")
        var = parser.transfer_2Darray(1, 3, 4, 12)

        self.assert_equal_arrays(var, numpy.array([
            [30.0,  34.8,  36.3,  36.1,  34.6,  32.0,  28.4,  23.9,  18.5,  12.2],
            [36.5,  41.3,  42.8,  42.6,  41.1,  38.5,  34.9,  30.4,  25.0,  18.7],
            [42.8,  47.6,  49.1,  48.9,  47.4,  44.8,  41.2,  36.7,  31.3,  25.0],
            [48.4,  53.1,  54.7,  54.5,  53.0,  50.4,  46.8,  42.3,  36.9,  30.6]
        ]))


class FileParserDelimFeature(unittest.TestCase):

    def setUp(self):
        from openmdao.utils.file_wrap import FileParser

        global parser  # global so we don't need `self.` in feature doc
        parser = FileParser()

        parser._data = [
            "CASE 1",
            "3,7,2,4,5,6"
        ]

    def test_parse_default_delim(self):
        parser.reset_anchor()
        parser.mark_anchor("CASE")
        var = parser.transfer_var(1, 2)

        self.assertEqual((var, type(var)), (",7,2,4,5,6", str))

    def test_parse_comma_delim(self):
        parser.reset_anchor()
        parser.mark_anchor("CASE")
        parser.set_delimiters(", ")
        var = parser.transfer_var(1, 2)

        self.assertEqual((var, type(var)), (7, int))


class FileParserColumnsFeature(unittest.TestCase):

    def setUp(self):
        from openmdao.utils.file_wrap import FileParser

        global parser  # global so we don't need `self.` in feature doc
        parser = FileParser()

        parser._data = [
            "CASE 1",
            "12345678901234567890",
            "TTF    3.7-9.4434967"
        ]

    def test_parse_columns(self):
        parser.reset_anchor()
        parser.mark_anchor("CASE")
        parser.set_delimiters("columns")

        var1 = parser.transfer_var(2, 3, 3)
        var2 = parser.transfer_var(2, 4, 10)
        var3 = parser.transfer_var(2, 11, 20)

        self.assertEqual((var1, var2, var3), ('F', 3.7, -9.4434967))


class FileParserArrayColumnsFeature(unittest.TestCase):

    def setUp(self):
        from openmdao.utils.file_wrap import FileParser

        global parser  # global so we don't need `self.` in feature doc
        parser = FileParser()

        parser._data = [
            "CASE 2",
            "123456789012345678901234567890",
            "NODE 11 22 33 COMMENT",
            "NODE 44 55 66 STUFF"
        ]

    def test_parse_columns(self):
        parser.reset_anchor()
        parser.mark_anchor("CASE 2")

        parser.set_delimiters("columns")
        var = parser.transfer_array(2, 6, 3, 13)

        parser.set_delimiters(" \t")

        assert_near_equal(var,
                         numpy.array([11., 22., 33., 44., 55., 66.]))


if __name__ == "__main__":
    unittest.main()
