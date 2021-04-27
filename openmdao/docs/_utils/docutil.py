"""
A collection of functions for modifying source code that is embeded into the Sphinx documentation.
"""

import sys
import os
import re
import tokenize
import importlib
import inspect
import subprocess
import tempfile
import unittest
import traceback
import ast

from docutils import nodes

from collections import namedtuple

from io import StringIO

from sphinx.errors import SphinxError
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.html5 import HTML5Translator
from redbaron import RedBaron

import html as cgiesc

from openmdao.utils.general_utils import printoptions

sqlite_file = 'feature_docs_unit_test_db.sqlite'    # name of the sqlite database file
table_name = 'feature_unit_tests'   # name of the table to be queried

_sub_runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_sub.py')


# an input block consists of a block of code and a tag that marks the end of any
# output from that code in the output stream (via inserted print('>>>>>#') statements)
InputBlock = namedtuple('InputBlock', 'code tag')


class skipped_or_failed_node(nodes.Element):
    pass


def visit_skipped_or_failed_node(self, node):
    pass


def depart_skipped_or_failed_node(self, node):
    if not isinstance(self, (HTMLTranslator, HTML5Translator)):
        self.body.append("output only available for HTML\n")
        return

    html = '<div class="cell border-box-sizing code_cell rendered"><div class="output"><div class="inner_cell"><div class="{}"><pre>{}</pre></div></div></div></div>'.format(node["kind"], node["text"])
    self.body.append(html)


class in_or_out_node(nodes.Element):
    pass


def visit_in_or_out_node(self, node):
    pass


def depart_in_or_out_node(self, node):
    """
    This function creates the formatting that sets up the look of the blocks.
    The look of the formatting is controlled by _theme/static/style.css
    """
    if not isinstance(self, (HTMLTranslator, HTML5Translator)):
        self.body.append("output only available for HTML\n")
        return
    if node["kind"] == "In":
        html = '<div class="highlight-python"><div class="highlight"><pre>{}</pre></div></div>'.format(node["text"])
    elif node["kind"] == "Out":
        html = '<div class="cell border-box-sizing code_cell rendered"><div class="output_area"><pre>{}</pre></div></div>'.format(node["text"])

    self.body.append(html)


def node_setup(app):
    app.add_node(skipped_or_failed_node, html=(visit_skipped_or_failed_node, depart_skipped_or_failed_node))
    app.add_node(in_or_out_node, html=(visit_in_or_out_node, depart_in_or_out_node))


def remove_docstrings(source):
    """
    Return 'source' minus docstrings.

    Parameters
    ----------
    source : str
        Original source code.

    Returns
    -------
    str
        Source with docstrings removed.
    """
    io_obj = StringIO(source)
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
    """
    Utility function for removing a node using RedBaron.

    RedBaron has some problems with modifying code lines that run across
    multiple lines. ( It is mentioned somewhere online but cannot seem to
    find it now. )

    RedBaron throws an Exception but when you check, it seems like it does
    what you asked it to do. So, for now, we ignore the Exception.
    """

    try:
        node.value.remove(node.value[index])
    except Exception as e:  # no choice but to catch the general Exception
        if str(e).startswith('It appears that you have indentation in your CommaList'):
            pass
        else:
            raise


def replace_asserts_with_prints(src):
    """
    Replace asserts with print statements.

    Using RedBaron, replace some assert calls with print statements that print the actual
    value given in the asserts. Depending on the calls, the actual value can be the first or second
    argument.

    Parameters
    ----------
    src : str
        String containing source lines.

    Returns
    -------
    str
        String containing source with asserts replaced by prints.
    """
    rb = RedBaron(src)  # convert to RedBaron internal structure

    # findAll is slow, so only check the ones that are present.
    base_assert = ['assertAlmostEqual', 'assertLess', 'assertGreater', 'assertEqual',
                   'assert_equal_arrays', 'assertTrue', 'assertFalse']
    used_assert = [item for item in base_assert if item in src]

    for assert_type in used_assert:
        assert_nodes = rb.findAll("NameNode", value=assert_type)
        for assert_node in assert_nodes:
            assert_node = assert_node.parent
            remove_redbaron_node(assert_node, 0)  # remove 'self' from the call
            assert_node.value[0].replace('print')
            if assert_type not in ['assertTrue', 'assertFalse']:
                # remove the expected value argument
                remove_redbaron_node(assert_node.value[1], 1)

    if 'assert_rel_error' in src:
        assert_nodes = rb.findAll("NameNode", value='assert_rel_error')
        for assert_node in assert_nodes:
            assert_node = assert_node.parent
            # If relative error tolerance is specified, there are 4 arguments
            if len(assert_node.value[1]) == 4:
                # remove the relative error tolerance
                remove_redbaron_node(assert_node.value[1], -1)
            remove_redbaron_node(assert_node.value[1], -1)  # remove the expected value
            # remove the first argument which is the TestCase
            remove_redbaron_node(assert_node.value[1], 0)
            #
            assert_node.value[0].replace("print")

    if 'assert_near_equal' in src:
        assert_nodes = rb.findAll("NameNode", value='assert_near_equal')
        for assert_node in assert_nodes:
            assert_node = assert_node.parent
            # If relative error tolerance is specified, there are 3 arguments
            if len(assert_node.value[1]) == 3:
                # remove the relative error tolerance
                remove_redbaron_node(assert_node.value[1], -1)
            remove_redbaron_node(assert_node.value[1], -1)  # remove the expected value
            assert_node.value[0].replace("print")

    if 'assert_almost_equal' in src:
        assert_nodes = rb.findAll("NameNode", value='assert_almost_equal')
        for assert_node in assert_nodes:
            assert_node = assert_node.parent
            # If relative error tolerance is specified, there are 3 arguments
            if len(assert_node.value[1]) == 3:
                # remove the relative error tolerance
                remove_redbaron_node(assert_node.value[1], -1)
            remove_redbaron_node(assert_node.value[1], -1)  # remove the expected value
            assert_node.value[0].replace("print")

    return rb.dumps()


def remove_initial_empty_lines(source):
    """
    Some initial empty lines were added to keep RedBaron happy.
    Need to strip these out before we pass the source code to the
    directive for including source code into feature doc files.
    """

    idx = re.search(r'\S', source, re.MULTILINE).start()
    return source[idx:]


def get_source_code(path):
    """
    Return source code as a text string.

    Parameters
    ----------
    path : str
        Path to a file, module, function, class, or class method.

    Returns
    -------
    str
        The source code.
    int
        Indentation level.
    module or None
        The imported module.
    class or None
        The class specified by path.
    method or None
        The class method specified by path.
    """

    indent = 0
    class_obj = None
    method_obj = None

    if path.endswith('.py'):
        if not os.path.isfile(path):
            raise SphinxError("Can't find file '%s' cwd='%s'" % (path, os.getcwd()))
        with open(path, 'r') as f:
            source = f.read()
        module = None
    else:
        # First, assume module path since we want to support loading a full module as well.
        try:
            module = importlib.import_module(path)
            source = inspect.getsource(module)

        except ImportError:

            # Second, assume class and see if it works
            try:
                parts = path.split('.')

                module_path = '.'.join(parts[:-1])
                module = importlib.import_module(module_path)
                class_name = parts[-1]
                class_obj = getattr(module, class_name)
                source = inspect.getsource(class_obj)
                indent = 1

            except ImportError:

                # else assume it is a path to a method
                module_path = '.'.join(parts[:-2])
                module = importlib.import_module(module_path)
                class_name = parts[-2]
                method_name = parts[-1]
                class_obj = getattr(module, class_name)
                method_obj = getattr(class_obj, method_name)
                source = inspect.getsource(method_obj)
                indent = 2

    return remove_leading_trailing_whitespace_lines(source), indent, module, class_obj, method_obj


def remove_raise_skip_tests(src):
    """
    Remove from the code any raise unittest.SkipTest lines since we don't want those in
    what the user sees.
    """
    rb = RedBaron(src)
    raise_nodes = rb.findAll("RaiseNode")
    for rn in raise_nodes:
        # only the raise for SkipTest
        if rn.value[:2].dumps() == 'unittestSkipTest':
            rn.parent.value.remove(rn)
    return rb.dumps()


def remove_leading_trailing_whitespace_lines(src):
    """
    Remove any leading or trailing whitespace lines.

    Parameters
    ----------
    src : str
        Input code.

    Returns
    -------
    str
        Code with trailing whitespace lines removed.
    """
    lines = src.splitlines()

    non_whitespace_lines = []
    for i, l in enumerate(lines):
        if l and not l.isspace():
            non_whitespace_lines.append(i)
    imin = min(non_whitespace_lines)
    imax = max(non_whitespace_lines)

    return '\n'.join(lines[imin: imax+1])


def is_output_node(node):
    """
    Determine whether a RedBaron node may be expected to generate output.

    Parameters
    ----------
    node : <Node>
        a RedBaron Node.

    Returns
    -------
    bool
        True if node may be expected to generate output, otherwise False.
    """
    if node.type == 'print':
        return True

    # lines with the following signatures and function names may generate output
    output_signatures = [
        ('name', 'name', 'call'),
        ('name', 'name', 'name', 'call')
    ]
    output_functions = [
        'setup', 'run_model', 'run_driver',
        'check_partials', 'check_totals',
        'list_inputs', 'list_outputs', 'list_problem_vars'
    ]

    if node.type == 'atomtrailers' and len(node.value) in (3, 4):
        sig = []
        for val in node.value:
            sig.append(val.type)
        func_name = node.value[-2].value
        if tuple(sig) in output_signatures and func_name in output_functions:
            return True

    return False


def split_source_into_input_blocks(src):
    """
    Split source into blocks; the splits occur at inserted prints.

    Parameters
    ----------
    src : str
        Input code.

    Returns
    -------
    list
        List of input code sections.
    """
    input_blocks = []
    current_block = []

    for line in src.splitlines():
        if 'print(">>>>>' in line:
            tag = line.split('"')[1]
            code = '\n'.join(current_block)
            input_blocks.append(InputBlock(code, tag))
            current_block = []
        else:
            current_block.append(line)

    if len(current_block) > 0:
        # final input block, with no associated output
        code = '\n'.join(current_block)
        input_blocks.append(InputBlock(code, ''))

    return input_blocks


def insert_output_start_stop_indicators(src):
    """
    Insert identifier strings so that output can be segregated from input.

    Parameters
    ----------
    src : str
        String containing input and output lines.

    Returns
    -------
    str
        String with output demarked.
    """
    lines = src.split('\n')
    print_producing = [
        'print(',
        '.setup(',
        '.run_model(',
        '.run_driver(',
        '.check_partials(',
        '.check_totals(',
        '.list_inputs(',
        '.list_outputs(',
        '.list_sources(',
        '.list_source_vars(',
        '.list_problem_vars(',
        '.list_cases(',
        '.list_model_options(',
        '.list_solver_options(',
    ]

    newlines = []
    input_block_number = 0
    in_try = False
    in_continuation = False
    head_indent = ''
    for line in lines:
        newlines.append(line)

        # Check if we are concluding a continuation line.
        if in_continuation:
            line = line.rstrip()
            if not (line.endswith(',') or line.endswith('\\') or line.endswith('(')):
                newlines.append('%sprint(">>>>>%d")' % (head_indent, input_block_number))
                input_block_number += 1
                in_continuation = False

        # Don't print if we are in a try block.
        if in_try:
            if 'except' in line:
                in_try = False
            continue

        if 'try:' in line:
            in_try = True
            continue

        # Searching for 'print(' is a little ambiguous.
        if 'set_solver_print(' in line:
            continue

        for item in print_producing:
            if item in line:
                indent = ' ' * (len(line) - len(line.lstrip()))

                # Line continuations are a litle tricky.
                line = line.rstrip()
                if line.endswith(',') or line.endswith('\\') or line.endswith('('):
                    in_continuation = True
                    head_indent = indent
                    break

                newlines.append('%sprint(">>>>>%d")' % (indent, input_block_number))
                input_block_number += 1
                break

    return '\n'.join(newlines)


def consolidate_input_blocks(input_blocks, output_blocks):
    """
    Merge any input blocks for which there is no corresponding output
    with subsequent blocks that do have output.

    Remove any leading and trailing blank lines from all input blocks.
    """
    new_input_blocks = []
    new_block = ''

    for (code, tag) in input_blocks:
        if tag not in output_blocks:
            # no output, add to new consolidated block
            if new_block and not new_block.endswith('\n'):
                new_block += '\n'
            new_block += code
        elif new_block:
            # add current input to new consolidated block and save
            if new_block and not new_block.endswith('\n'):
                new_block += '\n'
            new_block += code
            new_block = remove_leading_trailing_whitespace_lines(new_block)
            new_input_blocks.append(InputBlock(new_block, tag))
            new_block = ''
        else:
            # just strip leading/trailing from input block
            code = remove_leading_trailing_whitespace_lines(code)
            new_input_blocks.append(InputBlock(code, tag))

    # trailing input with no corresponding output
    if new_block:
        new_block = remove_leading_trailing_whitespace_lines(new_block)
        new_input_blocks.append(InputBlock(new_block, ''))

    return new_input_blocks


def extract_output_blocks(run_output):
    """
    Identify and extract outputs from source.

    Parameters
    ----------
    run_output : str or list of str
        Source code with outputs.

    Returns
    -------
    dict
        output blocks keyed on tags like ">>>>>4"
    """
    if isinstance(run_output, list):
        return sync_multi_output_blocks(run_output)

    output_blocks = {}
    output_block = None

    for line in run_output.splitlines():
        if output_block is None:
            output_block = []
        if line[:5] == '>>>>>':
            output = ('\n'.join(output_block)).strip()
            if output:
                output_blocks[line] = output
            output_block = None
        else:
            output_block.append(line)

    if output_block is not None:
        # It is possible to have trailing output
        # (e.g. if the last print_producing statement is in a try block)
        output_blocks['Trailing'] = output_block

    return output_blocks


def strip_decorators(src):
    """
    Remove any decorators from the source code of the method or function.

    Parameters
    ----------
    src : str
        Source code

    Returns
    -------
    str
        Source code minus any decorators
    """
    class Parser(ast.NodeVisitor):
        def __init__(self):
            self.function_node = None

        def visit_FunctionDef(self, node):
            self.function_node = node

        def get_function(self):
            return self.function_node

    tree = ast.parse(src)
    parser = Parser()
    parser.visit(tree)

    # get node for the first function
    function_node = parser.get_function()
    if not function_node.decorator_list:  # no decorators so no changes needed
        return src

    # Unfortunately, the ast library, for a decorated function, returns the line
    #   number for the first decorator when asking for the line number of the function
    # So using the line number for the argument for of the function, which is always
    #   correct. But we assume that the argument is on the same line as the function.
    # We also assume there IS an argument. If not, we raise an error.
    if function_node.args.args:
        function_lineno = function_node.args.args[0].lineno
    else:
        raise RuntimeError("Cannot determine line number for decorated function without args")
    lines = src.splitlines()

    undecorated_src = '\n'.join(lines[function_lineno - 1:])

    return undecorated_src


def strip_header(src):
    """
    Directly manipulating function text to strip header, usually or maybe always just the
    "def" lines for a method or function.

    This function assumes that the docstring and header, if any, have already been removed.

    Parameters
    ----------
    src : str
        source code
    """
    lines = src.split('\n')
    first_len = None
    for i, line in enumerate(lines):
        n1 = len(line)
        newline = line.lstrip()
        tab = n1 - len(newline)
        if first_len is None:
            first_len = tab
        elif n1 == 0:
            continue
        if tab != first_len:
            return '\n'.join(lines[i:])

    return ''


def dedent(src):
    """
    Directly manipulating function text to remove leading whitespace.

    Parameters
    ----------
    src : str
        source code
    """

    lines = src.split('\n')
    if lines:
        for i, line in enumerate(lines):
            lstrip = line.lstrip()
            if lstrip: # keep going if first line(s) are blank.
                tab = len(line) - len(lstrip)
                return '\n'.join(l[tab:] for l in lines[i:])
    return ''


def sync_multi_output_blocks(run_output):
    """
    Combine output from different procs into the same output blocks.

    Parameters
    ----------
    run_output : list of dict
        List of outputs from individual procs.

    Returns
    -------
    dict
        Synced output blocks from all procs.
    """
    if run_output:
        # for each proc's run output, get a dict of output blocks keyed by tag
        proc_output_blocks = [extract_output_blocks(outp) for outp in run_output]

        synced_blocks = {}

        for i, outp in enumerate(proc_output_blocks):
            for tag in outp:
                if str(outp[tag]).strip():
                    if tag in synced_blocks:
                        synced_blocks[tag] += "(rank %d) %s\n" % (i, outp[tag])
                    else:
                        synced_blocks[tag] = "(rank %d) %s\n" % (i, outp[tag])

        return synced_blocks
    else:
        return {}


def run_code(code_to_run, path, module=None, cls=None, shows_plot=False, imports_not_required=False):
    """
    Run the given code chunk and collect the output.
    """

    skipped = False
    failed = False

    if cls is None:
        use_mpi = False
    else:
        try:
            import mpi4py
        except ImportError:
            use_mpi = False
        else:
            N_PROCS = getattr(cls, 'N_PROCS', 1)
            use_mpi = N_PROCS > 1

    try:
        # use subprocess to run code to avoid any nasty interactions between codes

        # Move to the test directory in case there are files to read.
        save_dir = os.getcwd()

        if module is None:
            code_dir = os.path.dirname(os.path.abspath(path))
        else:
            code_dir = os.path.dirname(os.path.abspath(module.__file__))

        os.chdir(code_dir)

        if use_mpi:
            env = os.environ.copy()

            # output will be written to one file per process
            env['USE_PROC_FILES'] = '1'

            env['OPENMDAO_CURRENT_MODULE'] = module.__name__
            env['OPENMDAO_CODE_TO_RUN'] = code_to_run

            p = subprocess.Popen(['mpirun', '-n', str(N_PROCS), sys.executable, _sub_runner],
                                 env=env)
            p.wait()

            # extract output blocks from all output files & merge them
            output = []
            for i in range(N_PROCS):
                with open('%d.out' % i) as f:
                    output.append(f.read())
                os.remove('%d.out' % i)

        elif shows_plot:
            if module is None:
                # write code to a file so we can run it.
                fd, code_to_run_path = tempfile.mkstemp()
                with os.fdopen(fd, 'w') as tmp:
                    tmp.write(code_to_run)
                try:
                    p = subprocess.Popen([sys.executable, code_to_run_path],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ)
                    output, _ = p.communicate()
                    if p.returncode != 0:
                        failed = True

                finally:
                    os.remove(code_to_run_path)
            else:
                env = os.environ.copy()

                env['OPENMDAO_CURRENT_MODULE'] = module.__name__
                env['OPENMDAO_CODE_TO_RUN'] = code_to_run

                p = subprocess.Popen([sys.executable, _sub_runner],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
                output, _ = p.communicate()
                if p.returncode != 0:
                    failed = True

            output = output.decode('utf-8', 'ignore')
        else:
            # just exec() the code for serial tests.

            # capture all output
            stdout = sys.stdout
            stderr = sys.stderr
            strout = StringIO()
            sys.stdout = strout
            sys.stderr = strout

            # We need more precision from numpy
            with printoptions(precision=8):

                if module is None:
                    globals_dict = {
                        '__file__': path,
                        '__name__': '__main__',
                        '__package__': None,
                        '__cached__': None,
                    }
                else:
                    if imports_not_required:
                        # code does not need to include all imports
                        # Get from module
                        globals_dict = module.__dict__
                    else:
                        globals_dict = {}

                try:
                    exec(code_to_run, globals_dict)
                except Exception as err:
                    # for actual errors, print code (with line numbers) to facilitate debugging
                    if not isinstance(err, unittest.SkipTest):
                        for n, line in enumerate(code_to_run.split('\n')):
                            print('%4d: %s' % (n, line), file=stderr)
                    raise
                finally:
                    sys.stdout = stdout
                    sys.stderr = stderr

            output = strout.getvalue()

    except subprocess.CalledProcessError as e:
        output = e.output.decode('utf-8', 'ignore')
        # Get a traceback.
        if 'raise unittest.SkipTest' in output:
            reason_for_skip = output.splitlines()[-1][len('unittest.case.SkipTest: '):]
            output = reason_for_skip
            skipped = True
        else:
            output = "Running of embedded code {} in docs failed due to: \n\n{}".format(path, output)
            failed = True
    except unittest.SkipTest as skip:
        output = str(skip)
        skipped = True
    except Exception as exc:
        output = "Running of embedded code {} in docs failed due to: \n\n{}".format(path, traceback.format_exc())
        failed = True
    finally:
        os.chdir(save_dir)

    return skipped, failed, output


def get_skip_output_node(output):
    output = "Test skipped because " + output
    return skipped_or_failed_node(text=output, number=1, kind="skipped")


def get_interleaved_io_nodes(input_blocks, output_blocks):
    """
    Parameters
    ----------
    input_blocks : list of tuple
        Each tuple is a block of code and the tag marking it's output.

    output_blocks : dict
        Output blocks keyed on tag.
    """
    nodelist = []
    n = 1

    for (code, tag) in input_blocks:
        input_node = nodes.literal_block(code, code)
        input_node['language'] = 'python'
        nodelist.append(input_node)
        if tag and tag in output_blocks:
            outp = cgiesc.escape(output_blocks[tag])
            if (outp.strip()):
                output_node = in_or_out_node(kind="Out", number=n, text=outp)
                nodelist.append(output_node)
        n += 1

    if 'Trailing' in output_blocks:
        output_node = in_or_out_node(kind="Out", number=n, text=output_blocks['Trailing'])
        nodelist.append(output_node)

    return nodelist


def get_output_block_node(output_blocks):
    output_block = '\n'.join([cgiesc.escape(ob) for ob in output_blocks])
    return in_or_out_node(kind="Out", number=1, text=output_block)
