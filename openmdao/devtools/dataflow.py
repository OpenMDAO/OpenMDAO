"""
Various routines for data-flow analysis.
"""

from collections import defaultdict
from six import iteritems

import gast
from gast import NodeVisitor, FunctionDef, For, If, While, Try, Break, Continue, Assign, \
                 AugAssign, Name, Call, Attribute, Load, Store, NodeTransformer, Pass, walk
arguments = gast.gast.arguments
ast_ = gast
import networkx as nx

from openmdao.devtools.ast_tools import get_name

CONTROL_FLOW = (If, For, While, Break, Continue, Try)


class StatementInfo(object):
    __slots__ = ['defs', 'used', 'ins', 'outs']
    def __init__(self, defs, used):
        self.defs = defs
        self.used = used
        self.ins = None
        self.outs = None


class _BasicBlock(object):
    def __init__(self, id):
        self.id = id
        self.statements = []
        self.out = frozenset()

    def add_statement(self, statement, smap, subs=None):
        self.statements.append(statement)
        vis = UseDefVisitor()
        if subs:
            for s in subs:
                vis.visit(s)
        else:
            vis.visit(statement)

        smap[statement] = StatementInfo(vis.defs, vis.used)



def setup_reaching_def(block, smap, graph):
    inputs = []
    for p in graph.predecessors(block.id):
        inputs.extend(graph.nodes[p]['block'].out)

    before = hash(block.out)

    if block.statements:
        for i, s in enumerate(block.statements):
            sinfo = smap[s]
            defs = sinfo.defs
            if i > 0:
                inputs = out
            out = [(d, s) for d in defs]
            out.extend(def_ for def_ in inputs if def_[0] not in defs)
            sinfo.ins = inputs
            sinfo.outs = out
    else:
        out = inputs

    out = frozenset(out)
    changed = hash(out) != before

    block.out = out

    return changed


def reaching_definitions(cfg):
    nodes = cfg.graph.nodes
    blocks = [nodes['entry']['block']]
    blocks.extend(nodes[n]['block'] for _, n in nx.bfs_edges(cfg.graph, 'entry'))
    changed_set = set(blocks)
    while changed_set:
        for block in blocks:
            if block in changed_set:
                changed = setup_reaching_def(block, cfg.smap, cfg.graph)
                if not changed:
                    changed_set.remove(block)
                # print(block.id, [n for n, s in block.out])
        # print("CHANGED:", changed_set)


SKIP_REMOVE = (For, arguments)

def unused(cfg):
    graph = cfg.graph
    smap = cfg.smap
    used = []
    defs = set()

    for s, sinfo in iteritems(smap):
        defs.update((d, s) for d in sinfo.defs)
        used.extend(def_ for def_ in sinfo.ins if def_[0] in sinfo.used)

    # need a check against used_nodes for cases where an Assign has multiple
    # targets and at least one of them is used.
    used_nodes = set(d[1] for d in used)

    diff = defs.difference(used)
    diff = [def_ for def_ in diff if
            not isinstance(def_[1], SKIP_REMOVE) and
            def_[1] not in used_nodes]

    return diff


def remove_unused(topnode, check_stacks=True):
    for node in walk(topnode):
        if isinstance(node, FunctionDef):
            changed = True
            while changed:
                vis = ControlFlowGraphVisitor(node)
                #for n, data in vis.graph.nodes(data=True):
                    #block = data['block']
                    #statements = block.statements
                    #if statements:
                        #for s in statements:
                            #try:
                                #print(n, astunparse.unparse(s).strip())
                                ## if vis.smap[s].used: print('   USED:', vis.smap[s].used)
                                ## if vis.smap[s].defs: print('   DEFS:', vis.smap[s].defs)
                            #except:
                                #traceback.print_exc()
                                ## print(n, '???')
                    ## else:
                    ##     print(n, 'None')
                    #print("OUT:", [n for n, _ in block.out])

                #print("edges:", list(vis.graph.edges()))
                ppmap = get_matching_push_map(node) if check_stacks else {}

                un = unused(vis)
                changed = bool(un)
                import astunparse
                to_remove = set()
                #print("\nunused:")
                for n, s in un:
                    #print(n, astunparse.unparse(s))
                    to_remove.add(s)
                    if check_stacks and s in ppmap:
                        push = ppmap[s]
                        if push is None:
                            raise RuntimeError("No matching push for '%s'" %
                                               astunparse.unparse(s).strip())
                        else:
                            #print("matching push:", astunparse.unparse(ppmap[s]))
                            to_remove.add(ppmap[s])
                rem = Remover(to_remove)
                rem.visit(node)
                print("\n\nremoved:")
                # print(astunparse.unparse(node))

    return topnode

_pushpop = (
    'push',
    'push_stack',
    'pop',
    'pop_stack'
)


class StackFixer(NodeVisitor):
    def __init__(self):
        self.pushpop_map = defaultdict(lambda: [None, None])
        self.assign = None
        self.push = None
        self.pop = None

    def visit_Assign(self, node):
        self.assign = node
        if isinstance(node.value, Call):
            self.visit(node.value)
        self.assign = None

    def visit_Call(self, node):
        if (isinstance(node.func, Attribute) and node.func.attr in _pushpop and
                isinstance(node.func.value, Name) and node.func.value.id == 'tangent'):
            stack_id = node.args[-1].s
            if self.assign is not None:  # this is a pop
                self.pushpop_map[stack_id][1] = self.assign
            else:  # a push
                self.pushpop_map[stack_id][0] = node


def get_matching_push_map(node):
    matches = {}
    fixer = StackFixer()
    fixer.visit(node)
    for stack_id, lst in iteritems(fixer.pushpop_map):
        push, pop = lst

        matches[pop] = push
        # not sure if we need the following for anything
        # matches[push] = pop

    return matches

class Remover(NodeTransformer):
    def __init__(self, to_remove):
        self.to_remove = to_remove

    def _visit_statements(self, nodes):
        lst = []
        if nodes:
            for s in nodes:
                n = self.visit(s)
                if n:
                    lst.append(n)
        return lst

    def visit_Assign(self, node):  # targets, value
        if node in self.to_remove:
            return None
        return node

    def visit_AugAssign(self, node):  # target, value
        if node in self.to_remove:
            return None
        return node

    # def visit_Call(self, node):
    #     if node in self.to_remove:
    #         return None
    #     return node

    def visit_Expr(self, node):
        if node.value in self.to_remove:
            return None
        return node

    def visit_If(self, node):  # expr test, stmt* body, stmt* orelse
        node.body = self._visit_statements(node.body)
        if node.orelse:
            node.orelse = self._visit_statements(node.orelse)
        if not node.body and not node.orelse:
            return None
        return node

    def visit_For(self, node):  # expr target, expr iter, stmt* body, stmt* orelse
        node.body = self._visit_statements(node.body)
        if node.orelse:
            node.orelse = self._visit_statements(node.orelse)

        if node in self.to_remove and not node.body and not node.orelse:
            return None

        if not node.body:
            node.body = [Pass()]

        i = node.iter

        return node

    def visit_While(self, node):  # expr test, stmt* body, stmt* orelse
        node.body = self._visit_statements(node.body)
        if node.orelse:
            node.orelse = self._visit_statements(node.orelse)

        if not node.body and not node.orelse:
            return None
        return node


class ControlFlowGraphVisitor(NodeVisitor):
    """
    Construct a control flow graph from a FunctionDef AST node.

    Each statement is represented as a node. For control flow statements such
    as conditionals and loops the conditional itself is a node which either
    branches or cycles, respectively.
    """

    def __init__(self, node):
        if not isinstance(node, FunctionDef):
            raise TypeError('input must be a function definition')
        self.visit(node)

    def _add_block(self, parent=None, name=None):
        if name is None:
            b = _BasicBlock(self.nblocks)
            self.nblocks += 1
        else:
            b = _BasicBlock(name)
        self.graph.add_node(b.id, block=b)
        if parent is not None:
            self.graph.add_edge(parent.id, b.id)
        self.block = b
        return b

    def _end_loop(self):
        self.graph.add_edge(self.block.id, self.loop_stack[-1].id)
        self.loop_stack.pop()

    def _handle_breaks(self, loopblk):
        exitblk = self._add_block(self.block)
        for blk in loopblk.breaks:
            self.graph.add_edge(blk.id, exitblk.id)
        return exitblk

    def visit_statements(self, nodes):
        for node in nodes:
            if isinstance(node, CONTROL_FLOW):
                self.visit(node)
            else:
                self.block.add_statement(node, self.smap)
        return self.block

    def _remove_empty_blocks(self):
        g = self.graph
        to_remove = []
        for n, data in self.graph.nodes(data=True):
            block = data['block']
            if not block.statements and n != 'exit':
                for p in g.predecessors(block.id):
                    for s in g.successors(block.id):
                        g.add_edge(p, s)
                to_remove.append(block.id)

        g.remove_nodes_from(to_remove)

    def generic_visit(self, node):
        raise TypeError('unknown control flow:', type(node).__name__)

    def visit_FunctionDef(self, node):
        self.nblocks = 0
        self.smap = {}  # maps AST node to defs, used, and remove_count
        self.block = None
        self.loop_stack = []
        self.breaks = []
        self.graph = nx.DiGraph()
        exitblk = self._add_block(name='exit')
        self.block = None
        entryblk = self._add_block(name='entry')
        entryblk.add_statement(node.args, self.smap)
        self._add_block(entryblk)
        afterblk = self.visit_statements(node.body)
        self.graph.add_edge(afterblk.id, exitblk.id)
        self._remove_empty_blocks()
        self.block = None
        reaching_definitions(self)

    def visit_If(self, node):
        block = self.block
        block.add_statement(node, self.smap, subs=[node.test])
        self._add_block(block)
        after_body = self.visit_statements(node.body)
        exitblk = self._add_block(after_body)

        if node.orelse:
            self._add_block(block)
            after_orelse = self.visit_statements(node.orelse)
            self.graph.add_edge(after_orelse.id, exitblk.id)

        self.block = exitblk

    def visit_While(self, node):
        whblk = self._add_block(self.block)
        whblk.breaks = []
        self.loop_stack.append(whblk)
        whblk.add_statement(node, self.smap, subs=[node.test])
        self._add_block(whblk)
        afterblk = self.visit_statements(node.body)
        self._end_loop()

        if node.orelse:
            self._add_block(afterblk)
            afterelseblk = self.visit_statements(node.orelse)

        if whblk.breaks:
            self._handle_breaks(whblk)

    def visit_For(self, node):
        forblk = self._add_block(self.block)
        forblk.breaks = []
        forblk.add_statement(node, self.smap, subs=[node.target, node.iter])
        self.loop_stack.append(forblk)
        self.visit_statements(node.body)
        self._end_loop()

        if forblk.breaks or node.orelse:
            # if there are any breaks, create a loop exit block and point all of the breaks to that
            exitblk = self._handle_breaks(forblk)

        if node.orelse:
            elseblk = self._add_block(forblk)
            self.visit_statements(node.orelse, elseblk)
            self.graph.add_edge(self.block.id, exitblk.id)

    def visit_Try(self, node):
        tryblk = self._add_block(self.block)
        self.visit_statements(node.body, tryblk)

        hblks = []
        for handler in node.handlers:
            hblks.append(self._add_block(tryblk))
            self.visit_statements(handler.body)

        self._add_block(tryblk)
        self.visit_statements(node.orelse)

        finalblk = self._add_block(tryblk)
        self.visit_statements(node.finalbody)

        for hblk in hblks:
            self.graph.add_edge(hblk.id, finalblk.id)

    def visit_Break(self, node):
        self.loop_stack[-1].breaks.append(self.block)

    def visit_Continue(self, node):
        self.graph.add_edge(self.block.id, self.loop_stack[-1].id)


class UseDefVisitor(gast.NodeVisitor):
    """
    A visitor that collects all defines and usages of variables.
    """

    def __init__(self):
        super(UseDefVisitor, self).__init__()
        self.defs = set()
        self.used = set()
        self.aug = False

    def visit_Name(self, node):
        if node.ctx.__class__ is Load:
            self.used.add(node.id)
        else:
            self.defs.add(node.id)
            if self.aug:
                self.used.add(node.id)  # for AugAssign, target is defined AND used.

    def visit_Attribute(self, node):
        try:
            name = get_name(node)
        except TypeError:
            pass
        else:
            if node.ctx.__class__ is Load:
                self.used.add(name)
            else:
                self.defs.add(name)
                if self.aug:
                    self.used.add(name)  # for AugAssign, target is defined AND used.

    def visit_AugAssign(self, node):
        self.aug = True
        self.visit(node.target)
        self.aug = False
        self.visit(node.value)


if __name__ == '__main__':
    import sys
    import traceback
    import astunparse
    with open(sys.argv[1], 'r') as f:
        for node in ast_.walk(ast_.parse(f.read(), 'exec')):
            if isinstance(node, FunctionDef):
                print("\n\nFunction:", node.name)
                print("BEFORE:")
                print(astunparse.unparse(node))
                node = remove_unused(node, check_stacks=True)
                print("AFTER:")
                print(astunparse.unparse(node))

