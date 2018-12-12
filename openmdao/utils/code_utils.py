"""
Tools for working with code.
"""

import sys
import inspect
import ast
import textwrap
import importlib
from collections import defaultdict, OrderedDict
from six import iteritems, next

import networkx as nx


class OrderedDiGraph(nx.DiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict
    edge_attr_dict_factory = OrderedDict


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
    def __init__(self, class_):
        super(_SelfCallCollector, self).__init__()
        self.self_calls = defaultdict(list)
        self.class_ = class_

    def visit_Call(self, node):  # (func, args, keywords, starargs, kwargs)
        fncname = _get_long_name(node.func)
        class_ = self.class_
        if fncname is not None and fncname.startswith('self.') and len(fncname.split('.')) == 2:
            shortfnc = fncname.split('.')[1]
            if shortfnc not in self.self_calls[class_]:
                self.self_calls[class_].append(shortfnc)
            for arg in node.args:
                self.visit(arg)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            callnode = node.func.value
            n = _get_long_name(callnode.func)
            if n == 'super':  # this only works for a single call level
                sup_1 = _get_long_name(callnode.args[1])
                sup_0 = _get_long_name(callnode.args[0])
                if sup_1 == 'self' and sup_0 is not None and len(sup_0.split('.')) == 1:
                    mro = inspect.getmro(self.class_)
                    for i, c in enumerate(mro[:-1]):
                        if sup_0 == c.__name__:
                            # we need super of the current class
                            c = mro[i + 1]
                            fn = node.func.attr
                            if fn not in self.self_calls[c]:
                                self.self_calls[c].append(fn)
                            break
                    else:
                        self.generic_visit(node)
            else:
                self.generic_visit(node)
        else:
            self.generic_visit(node)


def _find_owning_class(mro, func_name):
    # TODO: this won't work for classes with __slots__

    for c in mro:
        if func_name in c.__dict__:
            return '.'.join((c.__name__, func_name)), c

    return None, None


def _get_nested_calls(starting_class, class_, func_name, parent, graph, seen):

    func = getattr(class_, func_name)
    src = inspect.getsource(func)
    dedented_src = textwrap.dedent(src)

    node = ast.parse(dedented_src, mode='exec')
    visitor = _SelfCallCollector(starting_class)
    visitor.visit(node)

    seen.add('.'.join((class_.__name__, func_name)))

    # now find the actual owning class for each call
    for klass, funcset in iteritems(visitor.self_calls):
        mro = inspect.getmro(klass)
        for f in funcset:
            full, c = _find_owning_class(mro, f)
            if full is not None:
                graph.add_edge(parent, full)
                if full not in seen:
                    _get_nested_calls(starting_class, klass, f, full, graph, seen)


def get_nested_calls(class_, func_name, stream=sys.stdout):
    graph = OrderedDiGraph()
    seen = set()

    full, klass = _find_owning_class(inspect.getmro(class_), func_name)
    if full is None:
        print("Can't find function '%s' in class '%s'." % (func_name, class_.__name__))
    else:
        graph.add_edge(None, full)
        parent = full
        _get_nested_calls(class_, klass, func_name, parent, graph, seen)

    if graph:
        seen = set([None])
        stack = [(0, iter(graph[None]))]
        while stack:
            depth, children = stack[-1]
            try:
                n = next(children)
                print("%s%s" % ('  ' * depth, n), file=stream)
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
    parser.add_argument('method_path', nargs=1, help='Full module path to desired class method.')
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        default='stdout', help='Output file.  Defaults to stdout.')


def _calltree_exec(options):
    """
    Process command line args and perform tracing on a specified python file.
    """
    parts = options.method_path[0].split('.')
    if len(parts) < 3:
        raise RuntimeError("You must supply the full module path to the function, "
                           "for example:  openmdao.api.Group._setup.")

    class_name = parts[-2]
    func_name = parts[-1]
    modpath = '.'.join(parts[:-2])

    mod = importlib.import_module(modpath)
    klass = getattr(mod, class_name)

    stream_map = { 'stdout': sys.stdout, 'stderr': sys.stderr}
    stream = stream_map.get(options.outfile)
    if stream is None:
        stream = open(options.outfile, 'w')

    get_nested_calls(klass, func_name, stream)
