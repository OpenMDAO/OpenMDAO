from __future__ import print_function

import sys
import ast
from ast import Assign, AugAssign, Name, Call, Store, Load, Str, Expr, Module
import astunparse
import inspect
import textwrap
import networkx as nx
from collections import defaultdict

class ImportScanner(ast.NodeVisitor):
    """
    This node visitor collects all import information from a file.
    """
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        """
        This executes every time an "import foo" style import statement
        is parsed.
        """
        for n in node.names:
            self.imports.append((n.name, n.asname, None))

    def visit_ImportFrom(self, node):
        """
        This executes every time a "from foo import bar" style import
        statement is parsed.
        """
        self.imports.append((node.module, None, [(n.name, n.asname) for n in node.names]))

    def get_import_lines(self):
        """
        Return source lines containing all of the imports.
        """
        lines = []
        for mod, alias, lst in self.imports:
            if lst is None:  # regular import
                if alias is None:
                    lines.append("import %s" % mod)
                else:
                    lines.append("import %s as %s" % (mod, alias))
            else:  # a 'from' import
                froms = []
                for imp, imp_alias in lst:
                    if imp_alias is None:
                        froms.append(imp)
                    else:
                        froms.append("%s as %s" % (imp, imp_allias))
                lines.append("from %s import %s" % (mod, ', '.join(froms)))

        return lines


def _get_attr_node(names, ctx):
    """Builds an Attribute node, or a Name node if names has just one entry."""
    node = ast.Name(id=names[0], ctx=ctx.__class__())
    for name in names[1:]:
        node = ast.Attribute(value=node, attr=name, ctx=ctx.__class__())
    return node


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


class DebugTransformer(ast.NodeTransformer):
    """
    Rewrite code to print the result of each Assign or AugAssign.
    """
    def __init__(self):
        super(DebugTransformer, self).__init__()
        self._stack = []
        self._funcs = []

    def _create_prints(self, targets):
        args = []
        indent = '   ' * len(self._stack)
        for t in targets:
            tstr = astunparse.unparse(t)[:-1]  # last char is '\n' so remove it
            n = ast.parse(tstr, mode='eval')
            args.extend((Str(indent + tstr + ' ='), n.body))

        return Expr(value=Call(func=Name(id='print', ctx=Load()), args=args, keywords=[]))

    def _visit_statements(self, statements):
        newstatements = []
        self._stack.append(newstatements)

        for s in statements:
            bn = self.visit(s)
            if bn is not None:
                newstatements.append(bn)

        return self._stack.pop()

    def visit_Assign(self, node):  # targets, value
        # don't need to visit the RHS

        self._stack[-1].append(node)  # add original assign
        self._stack[-1].append(self._create_prints(node.targets))  # add print statement

        return None  # return None so enclosing loop won't re-add this node to its list

    def visit_AugAssign(self, node):  # expr target, operator op, expr value
        # don't need to visit the RHS

        self._stack[-1].append(node)  # add original assign
        self._stack[-1].append(self._create_prints([node.target]))  # add print statement

        return None  # return None so enclosing loop won't re-add this node to its list

    def visit_FunctionDef(self, node):
        self._funcs.append(node)
        args = [Str('%s(' % node.name)]
        args.extend([Name(id=a.arg, ctx=Load()) for a in node.args.args if a.arg != 'self'])
        args.append(Str(')'))
        pnode = Expr(value=Call(func=Name(id='print', ctx=Load()), args=args, keywords=[]))
        node.body = [pnode] + self._visit_statements(node.body)
        return node

    def visit_For(self, node):  # expr target, expr iter, stmt* body, stmt* orelse
        node.body = [self._create_prints([node.target])] + self._visit_statements(node.body)
        if node.orelse:
            node.orelse = self._visit_statements(node.orelse)
        return node

    def visit_While(self, node):  # expr test, stmt* body, stmt* orelse
        node.body = self._visit_statements(node.body)
        if node.orelse:
            node.orelse = self._visit_statements(node.orelse)
        return node

    def visit_If(self, node):  # expr test, stmt* body, stmt* orelse
        node.body = self._visit_statements(node.body)
        if node.orelse:
            node.orelse = self._visit_statements(node.orelse)
        return node

    def visit_With(self, node):  # expr context_expr, expr? optional_vars, stmt* body
        node.body = self._visit_statements(node.body)
        return node

    def visit_Return(self, node):  #expr? value
        if node.value:
            indent = '   ' * len(self._stack)
            tstr = astunparse.unparse(node.value)[:-1]
            args = [Str(indent + 'return')]
            args.append(ast.parse(tstr, mode='eval').body)
            pnode = Expr(value=Call(func=Name(id='print', ctx=Load()), args=args, keywords=[]))

            self._stack[-1].append(pnode)  # add print
            self._stack[-1].append(node)  # add original return
            return None
        else:
            return node


def add_prints(stream=sys.stdout, verbose=0):
    """
    A decorator that dumps the result of each assignment in the wrapped function to stdout.
    """
    def pwrap(func):
        func_src = inspect.getsource(func)
        src = textwrap.dedent(func_src)

        # filter out this decorator from the source
        src = '\n'.join([l for l in src.splitlines() if not l.startswith('@add_prints')])
        if verbose > 1:
            print('SRC', file=stream)
            print(src, file=stream)

        f_ast = ast.parse(src, mode='exec')
        trans = DebugTransformer()
        trans.visit(f_ast)
        node = Module(body=[trans._funcs[0]])
        ast.fix_missing_locations(node)

        if verbose > 0:
            print('\nMODIFIED SRC', end='', file=stream)
            print(astunparse.unparse(node), file=stream)

        dec_code = compile(node, func.__code__.co_filename, mode='exec')
        exec(dec_code, func.__globals__)
        newfunc = func.__globals__[func.__name__]
        params = list(inspect.getargspec(func).args)

        def wrap(*args, **kwargs):
            save = sys.stdout
            sys.stdout = stream
            try:
                return newfunc(*args, **kwargs)
            finally:
                sys.stdout = save

        return wrap
    return pwrap

class StringSubscriptVisitor(ast.NodeVisitor):
    """
    A visitor that collects all string subscipted names so we can swap them out later.
    """

    def __init__(self):
        super(StringSubscriptVisitor, self).__init__()
        self.subscripts = defaultdict(list)

    def visit_Subscript(self, node):  # (value, slice, ctx)
        long_name = _get_long_name(node.value)
        if long_name is not None:
            if isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Str):
                self.subscripts[long_name].append(astunparse.unparse(node.slice).strip())
            else:
                self.visit(node.slice)
        else:
            self.generic_visit(node)


class DependencyVisitor(ast.NodeVisitor):
    """
    Perform dependency analysis on a function definition.
    """
    def __init__(self):
        super(DependencyVisitor, self).__init__()
        self.graph = nx.DiGraph()
        self.rhs_set = set()
        self.lhs_set = set()
        self.vset = None
        self.params = None
        self.calls = set()

    def visit_FunctionDef(self, node):
        self.params = set([n.arg for n in node.args.args])
        if node.args.vararg is not None:
            self.params.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            self.params.add(node.args.kwarg.arg)

        for bnode in node.body:
            self.visit(bnode)

    def visit_Call(self, node):  # (func, args, keywords, starargs, kwargs)
        long_name = _get_long_name(node.func)
        if long_name is not None:
            self.calls.add(long_name)
            for arg in node.args:
                self.visit(arg)
        else:
            self.generic_visit(node)

    def visit_Assign(self, node):  # (targets, value)
        self.rhs_set = set()
        self.lhs_set = set()
        self.vset = self.rhs_set
        self.visit(node.value)

        self.vset = self.lhs_set
        for target in node.targets:
            self.visit(target)

        self.vset = None

        for rhs in self.rhs_set:
            for lhs in self.lhs_set:
                self.graph.add_edge(lhs, rhs)

    def visit_AugAssign(self, node): # (target, op, value)
        self.rhs_set = set()
        self.lhs_set = set()
        self.vset = self.rhs_set
        self.visit(node.value)
        self.vset = self.lhs_set
        self.visit(node.target)
        self.vset = None

        for rhs in self.rhs_set:
            for lhs in self.lhs_set:
                self.graph.add_edge(lhs, rhs)

    def visit_Name(self, node):  # (id)
        if self.vset is not None:
            self.vset.add(node.id)

    def visit_Attribute(self, node):  # (value, attr)
        long_name = _get_long_name(node)
        if long_name is None:
            return self.generic_visit(node)
        if self.vset is not None:
            self.vset.add(long_name)

    def visit_Subscript(self, node):  # (value, slice, ctx)
        if self.vset is not None:
            long_name = _get_long_name(node.value)
            if long_name is not None:
                if isinstance(node.slice, ast.Index):
                    long_name = astunparse.unparse(node).strip()
                    self.vset.add(long_name)
                else:
                    self.vset.add(long_name)
                    self.visit(node.slice)
            else:
                self.generic_visit(node)
        else:
            self.generic_visit(node)

    def get_external(self):
        """
        Return a list of var names of vars not passed in or modified/created in the function.
        """
        g = self.graph
        return [node for node in g if node not in self.params and g.degree(node) == 1]


def dependency_analysis(src):
    f_ast = ast.parse(src, mode='exec')
    fdvis = DependencyVisitor()
    fdvis.visit(f_ast)

    # from openmdao.utils.graph_utils import all_connected_nodes
    # for node in fdvis.graph:
    #     print(node, list(set(list(all_connected_nodes(fdvis.graph, node))[1:])))

    return fdvis, f_ast


def get_external_vars(func_src):
    """
    Return names of variables not passed into or created/modified within the given function source.
    """
    f_ast = ast.parse(func_src, mode='exec')
    fdvis = DependencyVisitor()
    fdvis.visit(f_ast)
    return fdvis.get_external()


class NameTransformer(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping.copy()
        super(NameTransformer, self).__init__()

    def visit_Name(self, node):
        return ast.Name(id=self.mapping.get(node.id, node.id),
                        ctx=node.ctx.__class__())

    def visit_Attribute(self, node):
        long_name = _get_long_name(node)
        if long_name is None or long_name not in self.mapping:
            return self.generic_visit(node)
        return _get_attr_node(self.mapping[long_name].split('.'), node.ctx)

    def visit_Subscript(self, node):
        long_name = _get_long_name(node.value)
        xform = self.mapping.get(long_name)
        if xform is None:
            full_replace = True
            # look for transform of the full subscript, e.g.  foo['x']
            full = astunparse.unparse(node).rstrip()
            xform = self.mapping.get(full)
        else:
            full_replace = False

        if xform is not None:
            if full_replace:
                node = _get_attr_node(xform.split('.'), node.value.ctx)
            else:
                node.value = _get_attr_node(xform.split('.'), node.value.ctx)
            return node

        return super(NameTransformer, self).generic_visit(node)


def transform_ast_names(node, mapping):
    """
    Returns a new AST with the names transformed based on mapping.

    Note that this transforms only from the beginning of a name, so for example, if you have
    abc.xyz.abc and a mapping of { 'abc': 'XXX' }, you'll get 'XXX.xyz.abc', not 'XXX.xyz.XXX'.

    Parameters
    ----------
    node : ASTNode
        Top node of the original AST.
    mapping : dict
        Dict mapping original name to new name.

    Returns
    -------
    ASTNode
        Root of the transformed AST.
    """
    new_ast = NameTransformer(mapping).visit(node)
    ast.fix_missing_locations(new_ast)

    return new_ast


if __name__ == '__main__':
    mapping = {
        'thermo.Cp0': 'Thermo_Cp0',
        'thermo.H0': 'Thermo_H0',
        'thermo.S0': 'Thermo_S0',
    }

    funcsrc = """
def compute_foo(self, inputs, outputs):
    thermo = self.options['thermo']
    num_prod = thermo.num_prod
    num_element = thermo.num_element

    T = inputs['T']
    P = inputs['P']
    result_T = inputs['result_T']

    nj = inputs['n'][:num_prod]
    # nj[nj<0] = 1e-10 # ensure all concentrations stay non-zero
    n_moles = inputs['n_moles']

    self.dlnVqdlnP = dlnVqdlnP = -1 + inputs['result_P'][num_element]
    self.dlnVqdlnT = dlnVqdlnT = 1 - result_T[num_element]

    self.Cp0_T = Cp0_T = thermo.Cp0(T)
    Cpf = np.sum(nj*Cp0_T)

    self.H0_T = H0_T = thermo.H0(T)
    self.S0_T = S0_T = thermo.S0(T)
    self.nj_H0 = nj_H0 = nj*H0_T

    # Cpe = 0
    # for i in range(0, num_element):
    #     for j in range(0, num_prod):
    #         Cpe -= thermo.aij[i][j]*nj[j]*H0_T[j]*self.result_T[i]
    # vectorization of this for loop for speed
    Cpe = -np.sum(np.sum(thermo.aij*nj_H0, axis=1)*result_T[:num_element])
    Cpe += np.sum(nj_H0*H0_T)  # nj*H0_T**2
    Cpe -= np.sum(nj_H0)*result_T[num_element]

    outputs['h'] = np.sum(nj_H0)*R_UNIVERSAL_ENG*T

    try:
        val = (S0_T+np.log(n_moles/nj/(P/P_REF)))
    except FloatingPointError:
        P = 1e-5
        val = (S0_T+np.log(n_moles/nj/(P/P_REF)))


    outputs['S'] = R_UNIVERSAL_ENG * np.sum(nj*val)
    outputs['Cp'] = Cp = (Cpe+Cpf)*R_UNIVERSAL_ENG
    outputs['Cv'] = Cv = Cp + n_moles*R_UNIVERSAL_ENG*dlnVqdlnT**2/dlnVqdlnP

    outputs['gamma'] = -1*Cp/Cv/dlnVqdlnP
    outputs['rho'] = P/(n_moles*R_UNIVERSAL_SI*T)*100  # 1 Bar is 100 Kpa
    """

    global_ns = globals().copy()
    pre = set(global_ns)
    fdvis, f_ast = dependency_analysis(funcsrc)

    for c in fdvis.get_constants():
        print("constant:", c)

    # new_ast = transform_ast_names(orig_ast, mapping, global_ns)
    # cod = compile(new_ast, "<string>", mode='exec')
    # exec(cod, global_ns)
    # post = set(global_ns)

    # print("new stuff:", post - pre)
    # print(astunparse.unparse(new_ast))
