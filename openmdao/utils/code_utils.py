"""
Tools for working with code.
"""

import sys
import os
import inspect
import ast
import textwrap
import importlib
from collections import defaultdict, OrderedDict

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

    full, klass = _find_owning_class(inspect.getmro(class_), method_name)
    if full is None:
        print("Can't find function '%s' in class '%s'." % (method_name, class_.__name__))
    else:
        graph.add_edge(None, full)
        parent = full
        _get_nested_calls(class_, klass, method_name, parent, graph, seen)

    if graph and stream is not None:
        seen = set([None])
        stack = [(0, iter(graph[None]))]
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

    sys.path.append(os.getcwd())

    mod = importlib.import_module(modpath)
    klass = getattr(mod, class_name)

    stream_map = {'stdout': sys.stdout, 'stderr': sys.stderr}
    stream = stream_map.get(options.outfile)
    if stream is None:
        stream = open(options.outfile, 'w')

    get_nested_calls(klass, func_name, stream)


if __name__ == '__main__':
    import openmdao.api as om

    get_nested_calls(om.LinearBlockGS, 'solve')
