import sqlite3
import importlib
import inspect

import unittest


import cStringIO, tokenize


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
        ltext = tok[4]
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



sqlite_file = 'feature_docs_unit_test_db.sqlite'    # name of the sqlite database file
table_name = 'feature_tests'   # name of the table to be queried


def get_test_source_code_for_feature(feature_name):

    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()

    test_source_code_for_feature = []

    cur.execute('SELECT method_name FROM {tn} WHERE feature="{fn}"'.\
            format(tn=table_name, fn=feature_name))
    all_rows = cur.fetchall()
    for r in all_rows:
        method_path = r[0]
        print 'method_path', method_path
        # get lines that look like this
        # openmdao.tests.test_backtracking_copy.TestBackTrackingCopy.test_newton_with_backtracking_2
        # openmdao.tests.test_exec_comp.TestExecComp.test_complex_step
        module_path = '.'.join(method_path.split('.')[:-2])
        class_name = method_path.split('.')[-2]
        method_name = method_path.split('.')[-1]
        m = importlib.import_module(module_path)
        cls = getattr(m,class_name)
        meth = getattr(cls,method_name)
        # source = inspect.getsource( meth )
        source = inspect.getsource( meth )
        source_minus_docstrings = remove_docstrings(source)
        test_source_code_for_feature.append(source_minus_docstrings)
    return test_source_code_for_feature

if __name__ == "__main__":
    print get_test_source_code_for_feature('derivatives')
