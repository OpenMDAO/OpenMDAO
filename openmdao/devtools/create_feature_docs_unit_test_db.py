"""
  This script is to be called from the makefile that builds
  the OpenMDAO docs.

  It does the following:

    1. Open up a sqlite database in which to store the
        results of this script
    2. Discover all of the OpenMDAO unit tests
    3. For each unit test, check to see if the docstring
        has a Features section
    4. The arguments for that section list the features
        that are relevant to the unit test
    5. For each argument, store a record in the database
        with this information:
            method_path
            feature
            title

        The method_path is the dotted path to the
            unit test method,
            e.g. openmdao.tests.test_exec_comp.TestExecComp.test_complex_step

        The feature is the argument from the Feature section

        The title is the Short summary from the docstring of the unit test

    6. Close the database

    This database will be used later in the doc generating process
        as the feature docs will make use of a custom directive that
        includes the unit test code(s) into the feature doc automatically
"""

import cStringIO
import importlib
import inspect
import os
import re
import sqlite3
import subprocess
import tempfile
import textwrap
import tokenize
import unittest

from numpydoc.docscrape import NumpyDocString, Reader, ParseError
from numpydoc.docscrape_sphinx import SphinxDocString

from redbaron import RedBaron

# The results of this script will be stored in a sqlite database file
# sqlite_file = 'feature_docs_unit_test_db.sqlite'    # name of the sqlite database file
# table_name = 'feature_unit_tests'   # name of the table to be queried

from openmdao.docs.utils.get_test_source_code_for_feature import sqlite_file, table_name


# pylint: disable=C0103

# ------------------------begin monkeypatch-----------------------
# monkeypatch to let our docs include a section called "Features"
#    This new custom section will let unit test developers
#    indicate what features this unit test is relevant to.
# For example,
#
#        Features
#        --------
#        IndepVarComp
#        derivatives
#
# indicates that this unit test is relevant to any feature docs
# that discuss IndepVarComp or derivatives

def __str_SphinxDocString__(self, indent=0, func_role="obj"):
    out = []
    out += self._str_signature()
    out += self._str_index() + ['']
    out += self._str_summary()
    out += self._str_extended_summary()
    out += self._str_param_list('Args')
    out += self._str_options('Options')
    out += self._str_options('Params')
    out += self._str_returns()
    for param_list in ('Other Args', 'Raises', 'Warns'):
        out += self._str_param_list(param_list)
    out += self._str_warnings()
    out += self._str_see_also(func_role)
    out += self._str_section('Notes')
    out += self._str_section('Features')
    out += self._str_references()
    out += self._str_examples()
    for param_list in ('Attributes', 'Methods'):
        out += self._str_member_list(param_list)
    out = self._str_indent(out, indent)
    return '\n'.join(out)


def __init_NumpyDocString__(self, docstring, config={}):
    orig_docstring = docstring
    docstring = textwrap.dedent(docstring).split('\n')

    self._doc = Reader(docstring)
    self._parsed_data = {
        'Signature': '',
        'Summary': [''],
        'Extended Summary': [],
        'Parameters': [],
        'Returns': [],
        'Yields': [],
        'Raises': [],
        'Warns': [],
        'Other Parameters': [],
        'Attributes': [],
        'Methods': [],
        'See Also': [],
        'Notes': [],
        'Features': [],
        'Warnings': [],
        'References': '',
        'Examples': '',
        'index': {}
    }

    try:
        self._parse()
    except ParseError as e:
        e.docstring = orig_docstring
        raise


def __str_NumpyDocString__(self, func_role=''):
    out = []
    out += self._str_signature()
    out += self._str_summary()
    out += self._str_extended_summary()
    for param_list in ('Parameters', 'Returns', 'Yields',
                       'Other Parameters', 'Raises', 'Warns'):
        out += self._str_param_list(param_list)
    out += self._str_section('Warnings')
    out += self._str_see_also(func_role)
    for s in ('Notes', 'References', 'Examples', 'Features'):
        out += self._str_section(s)
    for param_list in ('Attributes', 'Methods'):
        out += self._str_param_list(param_list)
    out += self._str_index()
    return '\n'.join(out)


# Do the actual patch switchover to these local versions
NumpyDocString.__init__ = __init_NumpyDocString__
NumpyDocString.__str__ = __str_NumpyDocString__
SphinxDocString.__str__ = __str_SphinxDocString__
# --------------end monkeypatch---------------------


def remove_docstrings(source):
    """
    Returns 'source' minus docstrings.
    """
    io_obj = cStringIO.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        # ltext = tok[4] # in original code but not used here
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # This series of conditionals removes docstrings:
        if token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
                    # For example:
                    # def foo():
                    #     "The spaces before this docstring are tokenize.INDENT"
                    #     test = [
                    #         "The spaces before this string do not get a token"
                    #     ]
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out


def remove_redbaron_node(node, index):
    '''Utility function for removing a node using RedBaron.

    RedBaron has some problems with modifying code lines that run across
    multiple lines. ( It is mentioned somewhere online but cannot seem to
    find it now. )

    RedBaron throws an Exception but when you check, it seems like it does
    what you asked it to do. So, for now, we ignore the Exception.
    '''

    try:
        node.value.remove(node.value[index])
    except Exception as e:  # no choice but to catch the general Exception
        if str(e).startswith('It appears that you have indentation in your CommaList'):
            pass
        else:
            raise


def replace_asserts_with_prints(source_code):
    '''Using RedBaron, replace some assert calls with
    print statements that print the actual value given in the asserts.

    Depending on the calls, the actual value can be the first or second
    argument.'''

    rb = RedBaron(source_code)  # convert to RedBaron internal structure

    for assert_type in ['assertAlmostEqual', 'assertLess', 'assertGreater', 'assertEqual',
                        'assertEqualArrays', 'assertTrue', 'assertFalse']:
        assert_nodes = rb.findAll("NameNode", value=assert_type)
        for assert_node in assert_nodes:
            assert_node = assert_node.parent
            remove_redbaron_node(assert_node, 0)  # remove 'self' from the call
            assert_node.value[0].replace('print')
            if assert_type not in ['assertTrue', 'assertFalse']:
                remove_redbaron_node(assert_node.value[1], 1)  # remove the expected value argument

    assert_nodes = rb.findAll("NameNode", value='assert_rel_error')
    for assert_node in assert_nodes:
        assert_node = assert_node.parent
        remove_redbaron_node(assert_node.value[1], -1)  # remove the relative error tolerance
        remove_redbaron_node(assert_node.value[1], -1)  # remove the expected value
        remove_redbaron_node(assert_node.value[1], 0)  # remove the first argument which is
        #                                                  the TestCase
        assert_node.value[0].replace("print")

    source_code_with_prints = rb.dumps()  # get back the string representation of the code
    return source_code_with_prints


def get_method_body(method_code):
    '''Using the RedBaron module, get the body of a method.

    Do not want the definition signature line
    '''

    method_code = '\n' + method_code  # For some reason RedBaron has problems with this if
    #                                                     if it does not start with an empty line
    rb = RedBaron(method_code)
    def_node = rb.findAll("DefNode")[0]  # Look for the 'def' node. Should only be one!
    def_node.value.decrease_indentation(8)  # so that the code is all the way to the left
    return def_node.value.dumps()


def get_lines_before_test_cases(code):
    '''Only get the top part of the test file including imports and other class definitions

    Need this so the unit test has everything it needs to run
    '''

    top_code = ''
    rb = RedBaron(code)
    # Keep looping through the lines of code until encountering
    #   the start of a definition of a unittest.TestCase
    for r in rb:
        if r.type == 'string':
            continue
        if r.type == 'class':
            # check to see if any of the inherited classes are unittest.TestCase
            #   We do not want to stop if we hit a class definition that is not
            #   the definition of a TestCase, since that class might be needed in
            #   a test
            if 'unittest.TestCase' in r.inherit_from.dumps():
                break
        top_code += r.dumps() + '\n'
    return top_code


def remove_initial_empty_lines_from_source(source):
    '''Some initial empty lines were added to keep RedBaron happy.
    Need to strip these out before we pass the source code to the
    directive for including source code into feature doc files'''

    idx = re.search(r'\S', source, re.MULTILINE).start()
    return source[idx:]


def get_unit_test_source_and_run_outputs(method_path):
    module_path = '.'.join(method_path.split('.')[:-2])
    class_name = method_path.split('.')[-2]
    method_name = method_path.split('.')[-1]
    test_module = importlib.import_module(module_path)
    cls = getattr(test_module, class_name)
    meth = getattr(cls, method_name)
    source = inspect.getsource(meth)

    # Remove docstring from source code
    source_minus_docstrings = remove_docstrings(source)

    # We are using the RedBaron module in the next two function calls
    #    to get the code in the way we want it.

    # Only want the method body. Do not want the 'def' line
    method_body_source = get_method_body(source_minus_docstrings)

    # Replace some of the asserts with prints of the actual values
    source_minus_docstrings_with_prints = replace_asserts_with_prints(method_body_source)

    # Remove the initial empty lines
    source_minus_docstrings_with_prints_cleaned = remove_initial_empty_lines_from_source(
        source_minus_docstrings_with_prints)

    # Get all the pieces of code needed to run the unit test method
    module_source_code = inspect.getsource(test_module)
    lines_before_test_cases = get_lines_before_test_cases(module_source_code)
    setup_source_code = get_method_body(inspect.getsource(getattr(cls, 'setUp')))
    teardown_source_code = get_method_body(inspect.getsource(getattr(cls, 'tearDown')))
    code_to_run = '\n'.join([lines_before_test_cases,
                            setup_source_code,
                            source_minus_docstrings_with_prints_cleaned,
                            teardown_source_code])

    # Write it to a file so we can run it. Tried using exec but ran into problems with that
    with tempfile.NamedTemporaryFile(suffix='.py') as f:
        f.write(code_to_run)
        f.flush()
        run_outputs = subprocess.check_output(['python', f.name])

    return source_minus_docstrings_with_prints_cleaned, run_outputs


# remove any existing sqlite database file
try:
    os.remove(sqlite_file)
except OSError:
    pass

# Prepare to write to the sqlite database file
conn = sqlite3.connect(sqlite_file)
cur = conn.cursor()
cur.execute("CREATE TABLE {tn}(method_path TEXT, feature TEXT, title TEXT, \
                unit_test_source TEXT, run_outputs TEXT)".format(tn=table_name))

# Search for all the unit tests
test_loader = unittest.TestLoader()
suite = test_loader.discover('..', pattern="test_*.py")

for test_file in suite:  # Loop through the TestCases found
    for test_case in test_file:  # Loop though the test methods in the TestCases
        if isinstance(test_case, unittest.suite.TestSuite):
            for test_method in test_case:
                cls = test_method.__class__
                module_path = cls.__module__
                class_name = cls.__name__
                method_name = test_method._testMethodName
                method_path = '.'.join([module_path, class_name, method_name])
                test_doc = getattr(cls, method_name).__doc__
                if test_doc:
                    test_doc_numpy = NumpyDocString(test_doc)
                    if test_doc_numpy['Features']:
                        for feature in [f.strip() for f in test_doc_numpy['Features']]:
                            title = test_doc_numpy['Summary'][0]
                            unit_test_source, run_outputs = \
                                get_unit_test_source_and_run_outputs(method_path)
                            cur.execute(
                                'insert into {tn} values (?,?,?,?,?)'.format(tn=table_name),
                                (method_path, feature, title, unit_test_source, run_outputs))

conn.commit()
conn.close()
