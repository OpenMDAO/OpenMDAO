import unittest
import networkx as nx

from openmdao.utils.graph_utils import get_out_of_order_nodes

nodes = list(range(50))
orders = {i: i for i in nodes}

expected = [
    ([(1,2), (2,3), (3,4)],
     []),
    ([(1,2), (3,2), (3,4), (4,5), (5,6), (6,7), (9,7), (8,9)],
     [(3,2), (9,7)]),
    ([(1,2), (2,3), (3,4), (4,5), (5,3), (6,5), (6, 7), (7,8), (8,9)],
     [(6, 5)]),
    ([(1,2), (2,3), (3,4), (4,5), (5,6), (6, 2), (6,7), (7,8), (8,9)],
     []),
    ([(1,2), (2,3), (3,4), (4,5), (5, 3), (5,6), (6, 2), (6,7), (7,8), (8,9)],
     []),
    ([(1,2), (2,3), (3,4), (4, 1), (5, 4), (5,6), (6,7), (7, 5), (7,8), (8,9)],
     [(5, 4)]),
    ([(1,2), (2,3), (3,4), (3, 8), (4, 1), (5, 4), (5,6), (6,7), (7, 5), (7,8), (8,9)],
     [(5, 4)]),
    ([(1,2), (3,4), (4,5), (5,6), (6,7), (8,9)],
     []),
]


class GraphUtilsTestCase(unittest.TestCase):
    def test_out_of_order_nodes(self):
        for i in range(len(expected)):
            edges, expected_oo = expected[i]
            with self.subTest(f"edges {edges}"):
                graph = nx.DiGraph()
                graph.add_edges_from(edges)
                strongcomps, out_of_order = get_out_of_order_nodes(graph, orders)
                self.assertEqual(sorted(out_of_order), expected_oo)
