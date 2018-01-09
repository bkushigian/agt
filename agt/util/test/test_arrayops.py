from unittest import TestCase
from agt.util.arrayops import (zero, one, index, ones, diagonal,
                               dense_adjacency_matrix, generate_lower_triangle)


class TestArrayOps(TestCase):
    def test_index(self):
        l = list(range(10))
        for i in l:
            self.assertEqual(i, index(l, i))

    def test_ones(self):
        M = ones((3, 3))
        self.assertEqual((3,3), M.shape)
        for row in M:
            for x in row:
                self.assertEqual(one, x)

    def test_diagonal(self):
        M = diagonal(3)
        self.assertEqual((3, 3), M.shape)
        for i, row in enumerate(M):
            for j, x in enumerate(row):
                if i == j:
                    self.assertEqual(one, x)
                else:
                    self.assertEqual(zero, x)

    def test_dense_adjacency_matrix(self):
        M = dense_adjacency_matrix(3)
        for i, row in enumerate(M):
            for j, x in enumerate(row):
                if i == j:
                    self.assertEqual(0, x)
                else:
                    self.assertEqual(1, x)

    def generate_lower_triangle(self):
        indices = set(generate_lower_triangle(3))
        expected = {(1, 0), (2, 0), (2, 1)}
        self.assertEqual(expected, indices)

    def generate_lower_triangle_with_diagonal(self):
        indices = set(generate_lower_triangle(3, diagonal=True))
        expected = {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}
        self.assertEqual(expected, indices)

