import sqlite3
import importlib
import inspect
import unittest
import cStringIO, tokenize

from redbaron import RedBaron


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


def remove_redbaron_node(node,index):
    '''Utility function for removing a node using RedBaron.
    If the assert call is on multiple lines, RedBaron throws an Exception, but it actually
    still does what we want it to. Therefore we ignore that Exception. Multiple lines is a known
    problem with RedBaron according to some github issues.'''
    try:
        node.value.remove(node.value[index])
    except Exception as e:
        if str(e).startswith('It appears that you have indentation in your CommaList'):
            pass
        else:
            raise


def replace_asserts_with_prints(source_code):
    '''Using RedBaron, replace some assert statements with
    print statements that print the actual value in the asserts, which are
    assumed to be the first value. The expected is second'''

    rd3 = RedBaron(source_code) # convert to RedBaron internal structure

    for assert_type in ['assertAlmostEqual', 'assertLess', 'assertGreater', 'assertEqual', 'assertEqualArrays','assertTrue','assertFalse']:
        fa = rd3.findAll("NameNode", value=assert_type)
        for f in fa:
            assert_node = f.parent
            remove_redbaron_node(assert_node,0) # remove 'self' from the call
            assert_node.value[0].replace('print')
            if assert_type not in ['assertTrue','assertFalse']:
                remove_redbaron_node(assert_node.value[1],1) # remove the expected value argument

    fa = rd3.findAll("NameNode", value='assert_rel_error')
    for f in fa:
        assert_node = f.parent
        remove_redbaron_node(assert_node.value[1],-1) # remove the relative error tolerance
        remove_redbaron_node(assert_node.value[1],-1) # remove the expected value
        remove_redbaron_node(assert_node.value[1],0) # remove the first argument which is the TestCase
        assert_node.value[0].replace("print")

    source_code_with_prints = rd3.dumps() # get back the string representation of the code
    return source_code_with_prints

def get_method_body(method_code):
    '''
    Using RedBaron module, just get the method body. Do not want the
         definition signature line
    '''
    method_code = '\n' + method_code
    rbmc = RedBaron(method_code)
    def_node = rbmc.findAll("DefNode")[0] # Look for the 'def' node. Should only be one!
    def_node.value.decrease_indentation(8)
    return def_node.value.dumps()


def get_code_from_test_suite_method(test_suite_class, method_name):
    md = get_method_body(inspect.getsource( getattr(test_suite_class,method_name) ))
    return md

def get_top_lines(code):
    '''Only get the top part of the test file including imports and other class definitions'''

    # tc4[5].inherit_from[0]


    top_code = ''
    rbt = RedBaron(code)
    for r in rbt:
        if r.type == 'string':
            continue
        if r.type == 'class':
            # check to see if any of the inherited classes are unittest.TestCase
            if 'unittest.TestCase' in r.inherit_from.dumps() :
                    break
        top_code += r.dumps() + '\n'
    return top_code


sqlite_file = 'feature_docs_unit_test_db.sqlite'    # name of the sqlite database file
table_name = 'feature_tests'   # name of the table to be queried


def get_test_source_code_for_feature(feature_name):

    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()

    test_source_code_for_feature = []

    cur.execute('SELECT method_name, title FROM {tn} WHERE feature="{fn}"'.\
            format(tn=table_name, fn=feature_name))
    all_rows = cur.fetchall()
    for r in all_rows:
        method_path = r[0]
        title = r[1]
        # print 'method_path, title', method_path, title
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

        # Remove docstring from source code
        source_minus_docstrings = remove_docstrings(source)

        # We are using the RedBaron module in the next two function calls
        #    to get the code in the way we want it.
        # For some reason, RedBaron wants a blank first line. So we give it one
        source_minus_docstrings = '\n' + source_minus_docstrings

        # Only want the method body. Do not want the 'def' line
        method_body_source = get_method_body(source_minus_docstrings)

        # Replace some of the asserts with prints of the actual values
        source_minus_docstrings_with_prints = replace_asserts_with_prints(method_body_source)

        # Run the code
        module_source_code = inspect.getsource(m)
        top_lines = get_top_lines(module_source_code)
        setup_source_code = get_method_body(inspect.getsource( getattr(cls,'setUp') ))
        teardown_source_code = get_method_body(inspect.getsource( getattr(cls,'tearDown') ))

        code_to_run = top_lines + '\n' + setup_source_code + '\n' + source_minus_docstrings_with_prints + '\n' + teardown_source_code

        import sys
        import StringIO

        # print(70*'=')
        # print(code_to_run)
        # print(70*'=')
        # sys.exit()


        with open("run_test.py", 'w') as f:
            f.write(code_to_run)

        import subprocess
        run_outputs = subprocess.check_output(['python','run_test.py'])
        # print run_outputs


        # print "before StringIO"

        # create file-like string to capture output
        # codeOut = StringIO.StringIO()
        # codeErr = StringIO.StringIO()
        # print "after StringIO"

        # capture output and errors
        # sys.stdout = codeOut
        # print "after sys.stdout"
        # sys.stderr = codeErr

        # print "before run"

        # exec code_to_run

        # print "after run"

        # restore stdout and stderr
        # sys.stdout = sys.__stdout__
        # sys.stderr = sys.__stderr__

        # run_outputs = codeOut.getvalue()


        # run_outputs = \
        # '''
        # 3.14 88.2
        # 898.8 7768
        # '''
        # print('append',(title,source_minus_docstrings_with_prints,run_outputs))
        test_source_code_for_feature.append((title,source_minus_docstrings_with_prints,run_outputs))
        # print('len', len(test_source_code_for_feature))

    # print('RETURN', len(test_source_code_for_feature))
    return test_source_code_for_feature

if __name__ == "__main__":
    # print 'asfklsdhfklashdklfhkl'
    for scf in get_test_source_code_for_feature('derivatives'):
        # print(80*'-')
        print(scf)
