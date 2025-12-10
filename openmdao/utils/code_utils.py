"""
Tools for working with code.
"""

import sys
import os
import io
import inspect
import ast
import textwrap
import importlib
from types import LambdaType
from collections import defaultdict, OrderedDict
from tokenize import NAME, tokenize, untokenize
from openmdao.utils.om_warnings import issue_warning

import networkx as nx


def _get_long_name(node):
    # If the node is an Attribute or Name node that is composed
    # only of other Attribute or Name nodes, then return the full
    # dotted name for this node. Otherwise, i.e., if this node
    # contains Subscripts or Calls, return None.
    if isinstance(node, ast.Name):
        return node.id
    elif not isinstance(node, ast.Attribute):
        return None
    val = node.value
    parts = [node.attr]
    while True:
        if isinstance(val, ast.Attribute):
            parts.append(val.attr)
            val = val.value
        elif isinstance(val, ast.Name):
            parts.append(val.id)
            break
        else:  # it's more than just a simple dotted name
            return None
    return '.'.join(parts[::-1])


class _SelfCallCollector(ast.NodeVisitor):
    """
    An ast.NodeVisitor that records calls to self.* methods.
    """

    def __init__(self, class_):
        super().__init__()
        self.self_calls = defaultdict(list)
        self.class_ = class_
        self.mro = inspect.getmro(class_)
        self.mro_names = set([c.__name__ for c in self.mro])

    def visit_Call(self, node):  # (func, args, keywords, starargs, kwargs)
        fncname = _get_long_name(node.func)
        class_ = self.class_
        if fncname is not None:
            if fncname.startswith('self.') and len(fncname.split('.')) == 2:
                shortfnc = fncname.split('.')[1]
                if shortfnc not in self.self_calls[class_]:
                    self.self_calls[class_].append(shortfnc)
                for arg in node.args:
                    self.visit(arg)
            # check for Class.func(inst) form for base class method call
            elif (len(fncname.split('.')) == 2 and fncname.split('.')[0] in self.mro_names and
                  node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'self'):
                cname, func = fncname.split('.')
                for c in self.mro:
                    if c.__name__ == cname:
                        sub_mro = inspect.getmro(c)
                        for sub_c in sub_mro:
                            if func in sub_c.__dict__:
                                c = sub_c
                                break
                        if func not in self.self_calls[c]:
                            self.self_calls[c].append(func)
                        for arg in node.args:
                            self.visit(arg)
                        break
                else:
                    self.generic_visit(node)
            else:
                self.generic_visit(node)
        # check for super() call
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            callnode = node.func.value
            n = _get_long_name(callnode.func)
            # if this is a 'super' call, get the base of the specified class
            if n == 'super':  # this only works for a single call level
                if len(callnode.args) == 0:
                    sup_0 = self.mro[0].__name__
                    visit_super = True
                else:
                    sup_1 = _get_long_name(callnode.args[1])
                    sup_0 = _get_long_name(callnode.args[0])
                    visit_super = (sup_1 == 'self' and
                                   sup_0 is not None and len(sup_0.split('.')) == 1)

                if visit_super:
                    for i, c in enumerate(self.mro[:-1]):
                        if sup_0 == c.__name__:
                            # we need super of the specified class
                            sub_mro = inspect.getmro(c)
                            for sub_c in sub_mro:
                                if sub_c is not c:
                                    c = sub_c
                                    break
                            fn = node.func.attr
                            if fn not in self.self_calls[c]:
                                self.self_calls[c].append(fn)
                            for arg in node.args:
                                self.visit(arg)
                            break
                    else:
                        self.generic_visit(node)
            else:
                self.generic_visit(node)
        else:
            self.generic_visit(node)


def _find_owning_class(mro, func_name):
    """
    Return the full funcname and class where the function is first found in the class MRO.
    """
    # TODO: this won't work for classes with __slots__

    for c in mro:
        if func_name in c.__dict__:
            return '.'.join((c.__name__, func_name)), c

    return None, None


def _get_nested_calls(starting_class, class_, func_name, parent, graph, seen):
    """
    Parse the AST of the given method and all 'self' methods it calls and record owning classes.
    """
    func = getattr(class_, func_name)
    src = inspect.getsource(func)
    dedented_src = textwrap.dedent(src)

    node = ast.parse(dedented_src, mode='exec')
    visitor = _SelfCallCollector(starting_class)
    visitor.visit(node)

    seen.add('.'.join((class_.__name__, func_name)))

    # now find the actual owning class for each call
    for klass, funcset in visitor.self_calls.items():
        mro = inspect.getmro(klass)
        for f in funcset:
            full, c = _find_owning_class(mro, f)
            if full is not None:
                graph.add_edge(parent, full)
                if full not in seen:
                    _get_nested_calls(starting_class, c, f, full, graph, seen)


def get_nested_calls(class_, method_name, stream=sys.stdout):
    """
    Display the call tree for the specified class method and all 'self' class methods it calls.

    Parameters
    ----------
    class_ : class
        The starting class.
    method_name : str
        The name of the class method.
    stream : file-like
        The output stream where output will be displayed.

    Returns
    -------
    networkx.DiGraph
        A graph containing edges from methods to their sub-methods.
    """
    # moved this class def in here to keep the numpy doc scraper from barfing due to
    # stuff in nx.DiGraph.
    class OrderedDiGraph(nx.DiGraph):
        """
        A DiGraph using OrderedDicts for internal storage.
        """

        node_dict_factory = OrderedDict
        adjlist_dict_factory = OrderedDict
        edge_attr_dict_factory = OrderedDict

    graph = OrderedDiGraph()
    seen = set()
    top = object()

    full, klass = _find_owning_class(inspect.getmro(class_), method_name)
    if full is None:
        print("Can't find function '%s' in class '%s'." % (method_name, class_.__name__))
    else:
        graph.add_edge(top, full)
        parent = full
        _get_nested_calls(class_, klass, method_name, parent, graph, seen)

    if graph and stream is not None:
        seen = set([top])
        stack = [(0, iter(graph[top]))]
        while stack:
            depth, children = stack[-1]
            try:
                n = next(children)
                stream.write("%s%s\n" % ('  ' * depth, n))
                if n not in seen:
                    stack.append((depth + 1, iter(graph[n])))
                    seen.add(n)
            except StopIteration:
                stack.pop()

    return graph


def _calltree_setup_parser(parser):
    """
    Set up the command line options for the 'openmdao call_tree' command line tool.
    """
    parser.add_argument('method_path', nargs=1,
                        help='Full module path to desired class method, e.g., '
                        '"openmdao.components.exec_comp.ExecComp.setup".')
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        default='stdout', help='Output file.  Defaults to stdout.')


def _calltree_exec(options, user_args):
    """
    Process command line args and call get_nested_calls on the specified class method.
    """
    parts = options.method_path[0].split('.')
    if len(parts) < 3:
        raise RuntimeError("You must supply the full module path to the function, "
                           "for example:  openmdao.api.Group._setup.")

    class_name = parts[-2]
    func_name = parts[-1]
    modpath = '.'.join(parts[:-2])

    old_syspath = sys.path[:]
    sys.path.append(os.getcwd())

    try:
        mod = importlib.import_module(modpath)
        klass = getattr(mod, class_name)

        stream_map = {'stdout': sys.stdout, 'stderr': sys.stderr}
        stream = stream_map.get(options.outfile)
        if stream is None:
            stream = open(options.outfile, 'w')

        get_nested_calls(klass, func_name, stream)
    finally:
        sys.path = old_syspath


def _target_iter(targets):
    for target in targets:
        if isinstance(target, ast.Tuple):
            for t in target.elts:
                yield t
        else:
            yield target


class _AttrCollector(ast.NodeVisitor):
    """
    An ast.NodeVisitor that records class attribute names.
    """

    def __init__(self, class_dict):
        super().__init__()
        self.class_dict = class_dict
        self.class_stack = []
        self.func_stack = []
        self.names = None
        self.decnames = None

    def get_attributes(self):
        return self.class_dict

    def visit_ClassDef(self, node):
        full_name = '.'.join(self.class_stack[:] + [node.name])
        self.class_stack.append(full_name)
        self.class_dict[full_name] = set()
        for stmt in node.body:
            self.visit(stmt)
        self.class_stack.pop()

        if self.func_stack:  # ignore classes nested in functs
            del self.class_dict[full_name]

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self.func_stack.pop()

        if self.class_stack:
            # see if this is a property, and if so, treat as an attribute
            for dec in node.decorator_list:
                self.decnames = []
                self.visit(dec)
                if len(self.decnames) == 1 and self.decnames[0] == 'property':
                    self.class_dict[self.class_stack[-1]].add(node.name)

        self.decnames = None

    def visit_Assign(self, node):
        if self.class_stack:
            for t in _target_iter(node.targets):
                self.names = []
                self.visit(t)
                if len(self.names) > 1 and self.names[0] == 'self':
                    self.class_dict[self.class_stack[-1]].add(self.names[1])

            self.names = None

    def visit_Attribute(self, node):
        if self.names is not None:
            self.visit(node.value)
            self.names.append(node.attr)

    def visit_Name(self, node):
        if self.names is not None:
            self.names.append(node.id)
        elif self.decnames is not None:
            self.decnames.append(node.id)


def get_class_attributes(fname, class_dict=None):
    """
    Find all referenced attributes in all classes defined in the given file.

    Parameters
    ----------
    fname : str
        File name.
    class_dict : dict or None
        Dict mapping class names to attribute names.

    Returns
    -------
    dict
        The dict maps class name to a set of attribute names.
    """
    if class_dict is None:
        class_dict = {}

    with open(fname, 'r', encoding='utf-8') as f:
        source = f.read()
        node = ast.parse(source, mode='exec')
        visitor = _AttrCollector(class_dict)
        visitor.visit(node)
        return visitor.get_attributes()


def _get_return_name(node):
    return node.id if isinstance(node, ast.Name) else None


def _get_return_names(outs):
    """
    Return a list of (name or None) for each return value.

    If there are multiple returns that differ by name or number of return values, an exception
    will be raised.  If one entry in one return list has a name and another is None, the name
    will take precedence and no exception will be raised.

    Returns
    -------
    list
        The list of return names.  Some entries will be None if there was no simple name
        associated with a given return value.
    """
    if len(outs) == 0:
        return []
    if len(outs) == 1:
        return outs[0]

    names = outs[0].copy()
    length = len(names)
    for lst in outs[1:]:
        if len(lst) != length:
            raise RuntimeError("Function has multiple return statements with differing numbers "
                               "of return values.")

        for i, (name, newname) in enumerate(zip(names, lst)):
            if name is None:
                names[i] = newname
            elif newname is not None and name != newname:
                raise RuntimeError("Function has multiple return statements with different "
                                   f"return value names of {sorted((name, newname))} for "
                                   f"return value {i}.")
    return names


def get_return_names(func):
    """
    Return the names of the variables returned by the given function.

    Returns None for any return values that aren't a simple name.

    Parameters
    ----------
    func : function
        The function to be examined.

    Returns
    -------
    list
        The names of the variables returned by the given function.
    """
    class _FuncRetNameCollector(ast.NodeVisitor):
        """
        An ast.NodeVisitor that records return value names.

        Attributes
        ----------
        _ret_infos : list
            List containing one entry for each return statement, with each entry containing a list
            of name (or None) for each function return value.
        """

        def __init__(self, func):
            super().__init__()
            self._ret_infos = []
            self.fstack = []
            self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec'))

        def get_return_names(self):
            """
            Return a list of (name or None) for each return value.

            If there are multiple returns that differ by name or number of return values, an
            exception will be raised.  If one entry in one return list has a name and another is
            None, the name will take precedence and no exception will be raised.

            Returns
            -------
            list
                The list of return names.  Some entries will be None if there was no simple name
                associated with a given return value.
            """
            return _get_return_names(self._ret_infos)

        def visit_Return(self, node):
            """
            Visit a Return node.

            Parameters
            ----------
            node : ASTnode
                The return node being visited.
            """
            self._ret_infos.append([])

            if isinstance(node.value, ast.Tuple):
                for n in node.value.elts:
                    self._ret_infos[-1].append(_get_return_name(n))
            else:
                self._ret_infos[-1].append(_get_return_name(node.value))

        def visit_FunctionDef(self, node):
            """
            Visit a FunctionDef node.

            Parameters
            ----------
            node : ASTnode
                The function definition node being visited.
            """
            if self.fstack:
                return  # skip nested functions
            self.fstack.append(node)
            for stmt in node.body:
                self.visit(stmt)
            self.fstack.pop()

    return _FuncRetNameCollector(func).get_return_names()


class _FuncGrapher(ast.NodeVisitor):
    """
    An ast.NodeVisitor that builds a graph between a function's inputs and outputs.

    Note that this class assumes that all outputs of a called function are dependent on all inputs
    to that function, which may introduce dependencies that don't actually exist.
    """

    def __init__(self, node):
        super().__init__()
        self.rhs = []
        self.lhs = []
        self.names = None
        self.graph = nx.DiGraph()
        self.outs = []
        self.fstack = []
        self.visit(node)

    def _update_graph(self):
        for inp in self.rhs:
            for out in self.lhs:
                self.graph.add_edge(inp, out)
        self.lhs = []
        self.rhs = []

    def visit_FunctionDef(self, node):
        if self.fstack:
            # TODO: support nested functions
            raise RuntimeError("Function contains nested functions, which are not supported yet.")

        # add all input args to the graph
        for arg in node.args.args:
            if arg.arg != 'self':
                self.graph.add_node(arg.arg)

        self.fstack.append(node)
        for stmt in node.body:
            self.visit(stmt)
        self.fstack.pop()

    def visit_Assign(self, node):
        self.names = self.lhs
        for t in _target_iter(node.targets):
            self.visit(t)
        self.names = self.rhs
        self.visit(node.value)
        self.names = None
        self._update_graph()

    def visit_Attribute(self, node):
        name = _get_long_name(node.value)
        if name is not None:
            base = name.partition('.')[0]
            if base in self.graph:
                self.names.append(base)

    def visit_Name(self, node):
        if self.names is not None:
            self.names.append(node.id)

    def visit_Return(self, node):
        self.outs.append([])

        if isinstance(node.value, ast.Tuple):
            it = enumerate(node.value.elts)
        else:
            it = [(0, node.value)]

        for i, n in it:
            self.lhs = [f"@out{i}"]
            self.rhs = []
            self.names = self.rhs
            self.visit(n)
            self._update_graph()
            self.outs[-1].append(_get_return_name(n))


def get_func_graph(func, outnames=None, display=False):
    """
    Generate a graph between a function's inputs and outputs.

    Uses the AST to analyze the function and build a graph between inputs and outputs, so the
    function source must be available.

    Parameters
    ----------
    func : Callable
        The function to be analyzed.
    outnames : list or None
        The list of expected output variable names.
    display : bool
        If True, display the graph using pydot.

    Returns
    -------
    networkx.DiGraph or None
        A graph containing edges from inputs to outputs.  Returns None if the function graph
        couldn't be determined.
    """
    node = ast.parse(textwrap.dedent(inspect.getsource(func)), mode='exec')
    visitor = _FuncGrapher(node)

    retnames = _get_return_names(visitor.outs)
    inputs = set(inspect.signature(func).parameters)

    # check vs outnames
    if outnames is not None:
        if len(retnames) != len(outnames):
            raise RuntimeError("Number of return values in function does not match number of "
                               f"expected return names. ({outnames}) != ({retnames})")
        for ret, name in zip(retnames, outnames):
            if ret is not None and ret != name:
                raise RuntimeError(f"Return value name '{name}' in function does not match "
                                   f"expected name '{ret}.")
    else:
        outnames = []
        for ret in retnames:
            if ret is None or ret in inputs:
                outnames.append(f'out{len(outnames)}')
            else:
                outnames.append(ret)

    mapping = {f'@out{i}': name for i, name in enumerate(outnames)}
    visitor.graph = nx.relabel_nodes(visitor.graph, mapping)
    to_remove = [e for e in visitor.graph.edges() if e[0] == e[1]]
    visitor.graph.remove_edges_from(to_remove)
    # make sure all outputs exist as nodes in graph, even if they have no incoming edges
    visitor.graph.add_nodes_from(outnames)
    visitor.graph.graph['inputs'] = inputs
    visitor.graph.graph['outputs'] = outnames

    if display:
        # show the function graph visually
        from openmdao.visualization.graph_viewer import write_graph, _to_pydot_graph
        write_graph(_to_pydot_graph(visitor.graph), display=True)

    return visitor.graph


def get_function_deps(func, outputs=None, display=False):
    """
    Generate tuples of the form (output, input) for the given function.

    Only tuples where the output depends on the input are yielded.

    Note that currently the function grapher doesn't recurse into functions and assumes that all
    outputs of a called function are dependent on all inputs to that function. This may lead to
    some dependencies being reported that don't actually exist.

    Parameters
    ----------
    func : function or method with source available
        The function to be analyzed.
    outputs : list of str or None
        The list of output variable names.
    display : bool
        If True, display the function graph using pydot.

    Yields
    ------
    tuple
        A tuple of the form (output, input).
    """
    try:
        graph = get_func_graph(func, outputs, display)
        if graph is None:
            if outputs is None:
                return
    except Exception as err:
        # assume full dependency
        issue_warning(f"Can't determine function graph for function '{func.__name__}' "
                      f"so assuming all outputs depend on all inputs.  Error was: {err}")
        if outputs is None:
            yield '*', '*'
        else:
            for inp in inspect.signature(func).parameters:
                for out in outputs:
                    yield out, inp
        return

    outs = graph.graph['outputs']
    successors = graph.successors

    implicit = set(graph.graph['inputs']).intersection(graph.graph['outputs'])
    for imp in implicit:
        yield imp, imp

    for start in graph.graph['inputs']:
        visited = set([start])
        stack = [(start, successors(start))]
        while stack:
            _, succs = stack[-1]
            for succ in succs:
                if succ not in visited:
                    visited.add(succ)
                    if succ in outs:
                        yield succ, start
                    stack.append((succ, successors(succ)))
                    break
            else:
                stack.pop()


def block_filter(tokiter, blocks_to_remove, block_start_tok):
    """
    Remove blocks of code from a stream of tokens.

    Blocks are removed based on indentation level.  If a block's name matches one in
    blocks_to_remove, all non-blank lines where the first token is indented to a greater level
    than the block start token are removed.

    Parameters
    ----------
    tokiter : iterator
        Iterator of tokens.
    blocks_to_remove : set
        Set of block names to remove.
    block_start_tok : str
        The name of the block start token, e.g., 'def' or 'class'.

    Yields
    ------
    tuple
        The next token in the stream, unless it is part of a block that
        should be removed.
    """
    indent = None
    save = []
    for tok in tokiter:
        toktype, tokval, start, _, _ = tok
        tokcol = start[1]
        if save:  # we're on block start line after block start token
            if toktype == NAME and tokval not in blocks_to_remove:
                indent = None
                yield from save
                yield tok
            save = []
            continue
        elif toktype == NAME and tokval == block_start_tok:  # block start line
            indent = tokcol
            save.append(tok)  # we might need to emit this token if block doesn't match
            continue
        elif indent is not None:
            if tokcol > indent or not tokval.strip():  # skip lines that are indented or blank
                continue
            else:  # block is done
                indent = None

        yield tok


def find_block_start(srccode, block_name, block_start_tok):
    """
    Find the start of a block of code.

    Parameters
    ----------
    srccode : str
        Source code to search for block.
    block_name : str
        The name of the block to find.
    block_start_tok : str
        The name of the block start token, e.g., 'def' or 'class'.

    Returns
    -------
    tuple
        A tuple of the form (line number, column number, block start line) or (None, None, None) if
        the block was not found.
    """
    tokiter = tokenize(io.BytesIO(srccode.encode('utf-8')).readline)
    for tok in tokiter:
        toktype, tokval, start, _, _ = tok

        if toktype == NAME and tokval == block_start_tok:  # block start line
            try:
                nxt = next(tokiter)
            except StopIteration:
                return (None, None, None)
            ntoktype, ntokval, _, _, _ = nxt
            if ntoktype == NAME and ntokval == block_name:
                return start[0], start[1], tok[-1]

    return (None, None, None)


def remove_src_blocks(srccode, names, block_start_tok):
    """
    Remove blocks from a piece of source code.

    Parameters
    ----------
    srccode : str
        The source code.
    names : list of str
        List of blocks to be removed.
    block_start_tok : str
        The name of the block start token, e.g., 'def' or 'class'.

    Returns
    -------
    str
        The modified source code.
    """
    if not names:
        return srccode
    return untokenize(block_filter(tokenize(io.BytesIO(srccode.encode('utf-8')).readline),
                                   set(names), block_start_tok=block_start_tok)).decode()


def replace_src_block(srccode, block_name, new_block, block_start_tok):
    """
    Replace a block in a piece of source code.

    Parameters
    ----------
    srccode : str
        The source code.
    block_name : str
        The name of the block to be replaced.
    new_block : str
        The replacement block.
    block_start_tok : str
        The name of the block start token, e.g., 'def' or 'class'.

    Returns
    -------
    str
        The modified source code.
    """
    linenum, _, _ = find_block_start(srccode, block_name, block_start_tok)
    if linenum is None:
        raise RuntimeError(f"Block '{block_start_tok} {block_name}' not found in source code.")

    stream = io.StringIO(srccode)
    lines = srccode.splitlines()
    linenum -= 1  # i is zero indexed, so adjust linenum to be zero indexed
    for i, line in enumerate(lines):
        if i == linenum:
            # insert new block
            for subline in new_block.splitlines():
                print(subline, file=stream)
            print('', file=stream)
        print(line, file=stream)

    newsrc = stream.getvalue()
    # now remove the old block
    return untokenize(block_filter(tokenize(io.BytesIO(newsrc.encode('utf-8')).readline),
                                   {block_name}, block_start_tok=block_start_tok)).decode()


def is_lambda(f):
    """
    Return True if the given function is a lambda function.

    Parameters
    ----------
    f : function
        The function to check.

    Returns
    -------
    bool
        True if the given function is a lambda function.
    """
    return isinstance(f, LambdaType) and f.__name__ == "<lambda>"


class LambdaPickleWrapper(object):
    """
    A wrapper for a lambda function that allows it to be pickled.

    Parameters
    ----------
    lambda_func : function
        The lambda function to be wrapped.

    Attributes
    ----------
    _func : function
        The lambda function.
    _src : str
        The isolated source of the lambda function.
    """

    def __init__(self, lambda_func):
        """
        Initialize the wrapper.

        Parameters
        ----------
        lambda_func : function
            The lambda function to be wrapped.
        """
        self._func = lambda_func
        self._src = None

    def __call__(self, *args, **kwargs):
        """
        Call the lambda function.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        object
            The result of the lambda function.
        """
        return self._func(*args, **kwargs)

    def __getstate__(self):
        """
        Return the state of this object for pickling.

        The lambda function is converted to a string for pickling.

        Returns
        -------
        dict
            The state of this object.
        """
        state = self.__dict__.copy()
        state['_func'] = self._getsrc()
        return state

    def __setstate__(self, state):
        """
        Restore the state of this object after pickling.

        Parameters
        ----------
        state : dict
            The state of this object.
        """
        self.__dict__.update(state)
        self._func = eval(state['_func'])  # nosec

    def _getsrc(self):
        if self._src is None:
            self._src = _LambdaSrcFinder(self._func).src
            if self._src is None:
                raise RuntimeError("The fix for pickling lambda functions only works for python "
                                   "3.9 and above. Try updating to a newer python version or "
                                   "replacing the lambda with a regular function.")
        return self._src


class _LambdaSrcFinder(ast.NodeVisitor):
    """
    Given a lambda function, isolate the lambda function source from any surrounding code.
    """

    def __init__(self, func):
        super().__init__()
        self.src = None
        # note that inspect.getsource gives the source for the line that contains the lambda
        # function, so we have to isolate the lambda function itself
        self.visit(ast.parse(textwrap.dedent(inspect.getsource(func)), filename='<string>'))

    def visit_Lambda(self, node):
        if self.src is not None:
            # it's possible to have multiple lambdas defined on the same line, so raise an error
            # if we find more than one.
            raise RuntimeError("Only one lambda function is allowed per line when using "
                               "_LambdaSrcFinder.")
        try:
            self.src = ast.unparse(node)
        except AttributeError:
            # ast.unparse was added in python 3.9
            self.src = None


if __name__ == '__main__':
    import pprint
    pprint.pprint(get_class_attributes(__file__))
