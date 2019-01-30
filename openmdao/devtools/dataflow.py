"""
Various routines for data-flow analysis.
"""

from collections import defaultdict
from six import iteritems

import gast
from gast import NodeVisitor, FunctionDef, For, If, While, Try, Break, Continue, Assign, \
                 AugAssign, Name, Load, Store
ast_ = gast
import networkx as nx

from openmdao.devtools.ast_tools import get_name

CONTROL_FLOW = (If, For, While, Break, Continue, Try)


class _BasicBlock(object):
    def __init__(self, id):
        self.id = id
        self.statements = []
        self.out = {}

    def add_statement(self, statement, smap, subs=None):
        self.statements.append(statement)
        vis = UseDefVisitor()
        if subs:
            for s in subs:
                vis.visit(s)
        else:
            vis.visit(statement)

        #                 [defs, used, remove_count, global_index]
        smap[statement] = [vis.defs, vis.used, 0, len(smap)]



def setup_reaching_def(block, smap, graph):
    inputs = set()
    for p in graph.predecessors(block.id):
        inputs.update(graph.nodes[p]['block'].out)

    before = hash(block.out)

    out = set()
    if block.statements:
        for i, s in enumerate(block.statements):
            defs = smap[s][0]
            if i > 0:
                inputs = out
            kill = set(def_ for def_ in inputs if def_[0] in defs)
            gen = set((d, s) for d in defs)
            out = gen | (inputs - kill)
    else:
        out = inputs.copy()

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

    def visit_Name(self, node):
        if node.ctx.__class__ is Load:
            self.used.add(node.id)
        else:
            self.defs.add(node.id)

    def visit_Attribute(self, node):
        try:
            name = get_name(node)
        except TypeError:
            pass
        if node.ctx.__class__ is Load:
            self.used.add(name)
        else:
            self.defs.add(name)


if __name__ == '__main__':
    import sys
    import traceback
    import astunparse
    with open(sys.argv[1], 'r') as f:
        for node in ast_.walk(ast_.parse(f.read(), 'exec')):
            if isinstance(node, FunctionDef):
                print("\n\nFunction:", node.name)
                vis = ControlFlowGraphVisitor(node)
                for n, data in vis.graph.nodes(data=True):
                    block = data['block']
                    statements = block.statements
                    if statements:
                        for s in statements:
                            try:
                                print(n, astunparse.unparse(s).strip())
                                # if vis.smap[s][1]: print('   USED:', vis.smap[s][1])
                                # if vis.smap[s][0]: print('   DEFS:', vis.smap[s][0])
                            except:
                                traceback.print_exc()
                                # print(n, '???')
                    # else:
                    #     print(n, 'None')
                    # print("OUT:", [n for n, _ in block.out])

                print("edges:", list(vis.graph.edges()))