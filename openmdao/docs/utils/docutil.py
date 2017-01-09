"""
A collection of functions for modifying the source code
"""

import cStringIO
import re
import tokenize

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
