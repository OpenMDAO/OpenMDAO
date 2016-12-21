import unittest
import sqlite3

from numpydoc.docscrape_sphinx import SphinxDocString
from numpydoc.docscrape import NumpyDocString, Reader
import textwrap

#------------------------begin monkeypatch-----------------------
#monkeypatch to let our docs included a section called "Features" which we will then get the arguments from

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
        out += self._str_section('Features') #################################
        out += self._str_references()
        out += self._str_examples()
        for param_list in ('Attributes', 'Methods'):
            out += self._str_member_list(param_list)
        out = self._str_indent(out,indent)
        return '\n'.join(out)

def __init_NumpyDocString__(self, docstring, config={}):
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
            'Features': [], #####################
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
    for s in ('Notes', 'References', 'Examples', 'Features'):  ##############
        out += self._str_section(s)
    for param_list in ('Attributes', 'Methods'):
        out += self._str_param_list(param_list)
    out += self._str_index()
    return '\n'.join(out)

# Do the actual patch switchover to these local versions
NumpyDocString.__init__ = __init_NumpyDocString__
NumpyDocString.__str__ = __str_NumpyDocString__
SphinxDocString.__str__ = __str_SphinxDocString__
#--------------end monkeypatch---------------------

# The results of this script will be stored in a sqlite database file
sqlite_file = 'feature_docs_unit_test_db.sqlite'    # name of the sqlite database file
table_name = 'feature_unit_tests'   # name of the table to be queried

# remove any existing sqlite database file
import os
try:
    os.remove(sqlite_file)
except OSError:
    pass

# Prepare to write to the sqlite database file
conn = sqlite3.connect(sqlite_file)
cur = conn.cursor()
cur.execute("CREATE TABLE feature_tests(method_name TEXT, feature TEXT)")


# Search for all the unit tests
test_loader = unittest.TestLoader()
suite = test_loader.discover('..', pattern = "test_*.py")

for t in suite:
    # print type(t)
    for s in t:
        if isinstance(s, unittest.suite.TestSuite):
            for q in s:
                cls = q.__class__
                module_path = cls.__module__
                class_name = cls.__name__
                method_name = q._testMethodName
                method_path = '.'.join([module_path, class_name, method_name])
                test_doc = getattr(cls,method_name).__doc__
                if test_doc:
                    test_doc_numpy = NumpyDocString(test_doc)
                    if test_doc_numpy['Features']:
                        for feature in [ f.strip() for f in test_doc_numpy['Features']]:
                            # print 'qqq', method_path, feature
                            cur.execute('insert into feature_tests values (?,?)', (method_path,feature))

conn.commit()
conn.close()
