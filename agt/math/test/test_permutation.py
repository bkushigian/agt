from unittest import TestCase
from agt.math.permutations import permutation


class TestPermutation(TestCase):
    def test_multiply1(self):
        p1 = permutation(((1, 2, 3), (2, 3)))
        p2 = permutation(((2, 3), (3, 2, 1)))
        expected = permutation(tuple(tuple()))
        self.assertEqual(expected, p1 * p2)
        self.assertEqual(expected, p2 * p1)

    def test_inverse(self):
        p = permutation(((1, 2, 3),))
        q = p.inverse()
        expected = permutation()

    def test_acts_on(self):
        self.fail()

    def test_canonicalize(self):
        self.fail()
