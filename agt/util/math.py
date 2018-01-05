"""
This module stores various mathy operations to be used throughout this package.
"""
from math import factorial
from functools import reduce


def choose(n, k):
    """
    Compute nCk
    :param n:
    :param k:
    :return:
    """
    return factorial(n) / (factorial(k)*factorial(n - k))


def gcd(a, b):
    """
    Compute the greatest common divisor of a and b
    :param a:
    :param b:
    :return:
    """
    if a == 0:
        return b
    return gcd(b % a, a)


def lcm(a, b):
    """
    Compute the least common multiple of a and b
    :param a:
    :param b:
    :return:
    """

    if a == b == 0:
        return 0

    return (a * b) // gcd(a, b)


class Permutation:
    def __init__(self, cycles):
        orders = map(len, cycles)
        cycles = [c if isinstance(c, Cycle) else Cycle(c) for c in cycles]
        self.cycles = cycles
        self.order = reduce(lcm, orders, 1)
        self.map = {}
        self.inv = None

    def inverse(self):
        if self.inv is None:
            # TODO: Compute inverse
            cycles = self.__cycles[::-1]
            inversed = [x for x in map(lambda c: c.inverse(), cycles)]
            self.inv = Permutation(inversed)
        return self.inv

    def __call__(self, item):
        if item in self.map:
            return self.map[item]
        result = item
        for cycle in self.cycles[::-1]:
            result = cycle(result)
        self.map[item] = result
        return result

    def __mul__(self, other):
        """
        Computes self * other, which evaluates left to right
        :param other: other Permutation to evaluate
        :return: resulting permutation after composition
        """
        if not isinstance(other, Permutation):
            raise TypeError("Cannot compose permutation with non-permutation {}".format(other))
        return Permutation(self.cycles + other.cycles)


class Cycle(Permutation):
    def __init__(self, cycle):
        assert isinstance(cycle, tuple) or isinstance(cycle, list)
        if len(set(cycle)) != len(cycle):
            raise RuntimeError("Cycle {} has duplicate entry".format(cycle))
        self.cycle = cycle
        super().__init__([self])

    def inverse(self):
        """
        Compute the inverse of this cycle
        :return: inverse of self
        """
        if self.inv is None:
            self.inv = Cycle(self.__cycle[::-1])
            self.inv.__inverse = self

        return self.__inverse

    def __call__(self, item):
        if item in self.map:
            return self.map[item]

        result = item
        for i, elem in enumerate(self.cycle):
            if elem == item:
                result = self.cycle[(i + 1) % self.order]
                break
        self.map[item] = result
        return result

    def __len__(self):
        return len(self.cycle)

