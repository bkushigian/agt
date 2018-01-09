from unittest import TestCase
from math import inf

from agt.graph import MatrixGraph, Edge
from agt.util.arrayops import generate_lower_triangle
from agt.math.common import choose


class TestMatrixGraph(TestCase):

    def test_constructor1(self):
        g = MatrixGraph(4)
        self.assertEqual(4, g.order())
        self.assertEqual(0, g.size())

    def test_constructor2(self):
        g = MatrixGraph(order=4, edges=[(0, 1), (1, 2)])
        self.assertEqual(4, g.order())
        self.assertEqual(2, g.size())
        self.assertIn({0, 1}, g)
        self.assertIn({1, 2}, g)

    def test_constructor3(self):
        g = MatrixGraph(edges=[(0, 1), (1, 2), (2, 3)])
        self.assertEqual(4, g.order())
        self.assertEqual(3, g.size())
        self.assertIn({0, 1}, g)
        self.assertIn({1, 2}, g)
        self.assertIn({3, 2}, g)

    def test_constructor3(self):
        g = MatrixGraph(edges=[[1,2,3], [2,3], [], []])
        self.assertEqual(4, g.order())
        self.assertEqual(5, g.size())

    def test_add(self):
        g = MatrixGraph(4)
        self.assertFalse(g.is_edge(0, 1))
        g.add(0, 1)
        self.assertTrue(g.is_edge(0, 1))

    def test_adjacent(self):
        g = MatrixGraph(5).add(0, 1).add(1, 2).add(2, 3).add(3, 4)
        adj = g.adjacent(0)
        self.assertEqual(1, len(adj))
        self.assertIn(1, adj)

        adj = g.adjacent(1)
        self.assertEqual(2, len(adj))
        self.assertIn(0, adj)
        self.assertIn(2, adj)

        adj = g.adjacent(2)
        self.assertEqual(2, len(adj))
        self.assertIn(1, adj)
        self.assertIn(3, adj)

        adj = g.adjacent(3)
        self.assertEqual(2, len(adj))
        self.assertIn(2, adj)
        self.assertIn(4, adj)

        adj = g.adjacent(4)
        self.assertEqual(1, len(adj))
        self.assertIn(3, adj)

    def test_complement(self):
        g = MatrixGraph(4)
        g.add(0, 1)
        g.add(1, 2)
        g.add(2, 3)
        g.add(3, 0)
        c = g.complement()

        self.assertIn((0, 2), c)
        self.assertIn((1, 3), c)
        self.assertNotIn((0, 1), c)
        self.assertNotIn((1, 2), c)
        self.assertNotIn((2, 3), c)
        self.assertNotIn((3, 0), c)

    def test_connected1(self):
        g = MatrixGraph(4, [(0,1), (2,3)])
        self.assertFalse(g.connected())

    def test_connected2(self):
        g = MatrixGraph(4, [(0,1), (1,2), (2,3), (3,0)])
        self.assertTrue(g.connected())

    def test_connected3(self):
        g = MatrixGraph(5, [(0,1), (0,2), (0,3), (0,4)])
        self.assertTrue(g.connected())

    def test_create_random(self):
        total_edges = 0
        density = 0.5
        test_runs = 10
        for i in range(test_runs):
            g = MatrixGraph.create_random(32, density=density)
            for _,_ in g.edges():
                total_edges += 1

        expected = test_runs * choose(32, 2) * density
        actual = total_edges
        self.assertLessEqual(expected * 0.9, actual)
        self.assertLessEqual(actual, expected * 1.10)

    def test_d(self):
        """
        Test the degree function
        """
        g = MatrixGraph(order=4).add(0, 1).add(1, 2).add(2, 3).add(3, 0)
        self.assertEqual(2, g.d(0))
        self.assertEqual(2, g.d(1))
        self.assertEqual(2, g.d(2))
        self.assertEqual(2, g.d(3))
        g.add(0,2)
        self.assertEqual(3, g.d(0))
        self.assertEqual(2, g.d(1))
        self.assertEqual(3, g.d(2))
        self.assertEqual(2, g.d(3))

    def test_density1(self):
        g = MatrixGraph(10).add(0, 1).add(1, 2).add(2, 3).add(3, 4).add(4, 5)
        self.assertEqual(5/choose(10, 2), g.density())

    def test_density2(self):
        g = MatrixGraph(10)
        self.assertEqual(0, g.density())

    def test_density3(self):
        g = MatrixGraph(4).add(0, 1).add(0, 2).add(0, 3).add(1, 2).add(1, 3).add(2, 3)
        self.assertEqual(1.0, g.density())

    def test_density_random(self):
        for i in range(10):
            g = MatrixGraph.create_random(order=10)
            expected = g.size() / (g.size() + g.complement().size())
            self.assertEqual(expected, g.density())

    def test_directed(self):
        self.assertFalse(MatrixGraph(10).directed())

    def test_distance1(self):
        g = MatrixGraph(order=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(0, g.distance(0, 0))
        self.assertEqual(0, g.distance(1, 1))
        self.assertEqual(0, g.distance(2, 2))
        self.assertEqual(0, g.distance(3, 3))
        self.assertEqual(0, g.distance(4, 4))

        self.assertEqual(1, g.distance(0, 1))
        self.assertEqual(1, g.distance(1, 2))
        self.assertEqual(1, g.distance(2, 3))
        self.assertEqual(1, g.distance(3, 4))
        self.assertEqual(1, g.distance(1, 0))
        self.assertEqual(1, g.distance(2, 1))
        self.assertEqual(1, g.distance(3, 2))
        self.assertEqual(1, g.distance(4, 3))

        self.assertEqual(2, g.distance(0, 2))
        self.assertEqual(2, g.distance(1, 3))
        self.assertEqual(2, g.distance(2, 4))
        self.assertEqual(2, g.distance(4, 2))
        self.assertEqual(2, g.distance(3, 1))
        self.assertEqual(2, g.distance(2, 0))

    def test_distance2(self):
        g = MatrixGraph(order=5).add(0, 1).add(3, 4)
        self.assertEqual(inf, g.distance(0,4))

    def test_E(self):
        g = MatrixGraph(5)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for a,b in edges:
            g.add(a,b)
        for a,b in generate_lower_triangle(5, diagonal=True):
            if (a, b) in g:
                self.assertTrue(g.is_edge(a, b))
                self.assertTrue(g.is_edge(b, a))
            else:
                self.assertFalse(g.is_edge(a, b))
                self.assertFalse(g.is_edge(b, a))

    def test_edges(self):
        g = MatrixGraph(5)
        edges_in = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for (a, b) in edges_in:
            g.add(a, b)

        counter = 0
        for a, b in g.edges():
            self.assertTrue((a, b) in edges_in or (b, a) in edges_in)
            counter += 1
        self.assertEqual(len(edges_in), counter)

    def test_intersection(self):
        G = MatrixGraph(order=10).add(0, 1).add(2, 3).add(4, 5).add(8, 9)
        H = MatrixGraph(order=5).add(0, 1).add(1, 2).add(2, 3)
        GnH = G.intersection(H)
        self.assertEqual(5, GnH.order())
        self.assertEqual(2, GnH.size())
        edges = GnH.E()
        self.assertIn(Edge(0, 1), GnH)
        self.assertIn(Edge(2, 3), GnH)

    def test_remove(self):
        g = MatrixGraph(3)
        g.add(0, 1)
        g.remove(0, 1)
        self.assertNotIn((0, 1), g)
        self.assertNotIn((1, 0), g)

    def test_order(self):
        g = MatrixGraph(3)
        self.assertEqual(3, g.order())
        g = MatrixGraph(10)
        self.assertEqual(10, g.order())

    def test_iter(self):
        g = MatrixGraph(3).add(0, 1).add(0, 2).add(1, 2)
        assert len(list(g)) == 3
        l = list(g)
        self.assertIn(Edge(0, 1), l)
        self.assertIn(Edge(0, 2), l)
        self.assertIn(Edge(1, 2), l)

