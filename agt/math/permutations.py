"""
    agt.permutations
    ~~~~~~~~~~~~~~~~

    A simple library for dealing with primitive cycles (tuples with unique
    elements) and permutations (tuples/lists of cycles). The Cycle and
    Permutation classes are also defined here, although these can (should) be
    bussed out to SymPy.

    Permutation and Cycle classes:
        Technically a Cycle is a type of Permutation so it subclasses
        Permutation. However, Permutations are formed from Cycles and this
        creates some difficulties making everything play nicely. Rather than
        coming up with a clever/bug-prone solution I have factored out the
        low-level cycle/permutation stuff into some basic functions and use
        a PermutationFactory to create the permutations. As such, it is advised
        to not directly invoke the Permutation or Cycle constructors.
"""
from functools import reduce

import numpy as np

from math import inf
from math.common import lcm


def canonicalize_cycle_tuple(c):
    """
    Given a tuple (x1, ..., xn), check that it is valid as a cycle (i.e., no
    repeated entries) and return the canonicalized version of that tuple (i.e.,
    the minimum value is first).
    :param c: tuple to canonicalize
    :return: canonicalized version of c
    """

    assert isinstance(c, tuple)
    assert len(c) == len(set(c))  # Make sure there are no repeated items
    if len(c) < 2:
        return c
    i = c.index(min(c))   # Index of min
    result = rotate(c, i)


def canonicalize_permutation_tuple(cs):
    """
    Given a permutation (c1, c2, ..., ck) where each ci is a tuple representing
    a cycle, compute their canonical form. This includes:
        1. Separating into disjoint cycles (this may mean combining some of the
           ci
        2. Rotating each cycle (using canonicalize_cycle_tuple above)
        3. Arrange the canonicalized tuples in ascending order, ordered by their
           first element.

    :param cs: List of tuples to canonicalize
    :return: Canonicalized permutation as a tuple of tuples
    """
    if len(cs) < 2:
        return cs
    return sorted(tuple(canonicalize_cycle_tuple(c) for c in cs))


def find_cycle(f, initial, max_len=inf, fail_quietly=False):
    """
    Given a function f, generate the values f(i), f^2(i), ..., f^n(i) where
    f^{n+1}(i) is the initial value. This will go indefinitely or until
    max_len steps have been reached.

    :param f: Function to iterate
    :param initial: Initial input value to start the cycle
    :param max_len: Maximum length of the cycle to search for. Default is
    math.inf. If max_len is reached a RuntimeError is thrown unless fail_quietly
    is set to True.
    :param fail_quietly: if this is set to True then when max_len is reached
    the resulting generator is returned and no error is thrown.
    :return: a generator of values
    """
    yield initial
    i = f(initial)
    steps = 0
    while i != initial:
        if steps >= max_len:
            if fail_quietly:
                break
            else:
                raise RuntimeError("Cycle generation reached maximum steps: {}".format(max_len))
        steps += 1
        yield i
        i = f(i)


def rotate(seq, n):
    """
    Rotate seq (x1,x2,...,xk) to (xn, ..., xk, x1, ..., xn-1)
    :param seq: Sequence to rotate
    :param n: index to rotate by
    :return: Rotated sequence
    """

    l = len(seq)
    if l is 0:
        return seq
    N = n % l
    return seq[N:] + seq[:N]


def cycle_as_dict(cycle):
    """
    Given a cycle (x1, x2, ..., xn) return dictionary {x1:x2, x2:x3, ..., xn:x1}
    :param cycle:
    :return:
    """
    n = len(cycle)
    d = {}
    for i, elem in enumerate(cycle):
        d[elem] = cycle[(i + 1) % n]

    return d


def cycle_as_function(cycle):
    d = cycle_as_dict(cycle)

    def f(x):
        if x in d:
            return d[x]
        return x

    return f


def support(p):
    """
    Calculate the support of a permutation (the _support_ is the complement of the
    fixed points of an action).
    :param p: permutation
    :return: the support of `p`
    """
    result = set()

    # Some edge cases
    if not p:
        return result
    if not isinstance(p[0], tuple):
        return set(p)   # Is a cycle so just get its elements
    for c in p:
        result.update(c)
    return result


def permutation(*cycles):
    """
    Create a new permutation from tuples.
    :param cycles: a variadic argument of tuples representing cycles
    :return:
    """

    for c in cycles:
        if not isinstance(c, tuple):
            raise RuntimeError("Non-tuple passed as a cycle: {} of type {}".format(c, type(c)))

    # Now we sanitize the horrible stuff that will be passed to us
    if len(cycles) is 0:
        return Cycle(())

    f_cycles = [cycle_as_function(c) for c in cycles][::-1]

    def f(a):   # A helper function that composes our cycles for us
        r = a
        for g in f_cycles:
            r = g(r)
        return r

    s = sorted(support(cycles))
    visited = set()

    result = []
    for x in s:
        if x not in visited:
            c = tuple(find_cycle(f, x))
            visited.update(c)
            if len(c) > 1:
                result.append(c)

    if len(result) == 1:
        return Cycle(result[0])
    if len(result) == 0:
        return Cycle(())
    return Permutation(sorted(result))


class Permutation:
    def __init__(self, cycles):
        orders = map(len, cycles)
        self.cycles = tuple(cycles)
        self.order = reduce(lcm, orders, 1)
        self.map = {}
        self.inv = None

    def inverse(self):
        if self.inv is None:
            # TODO: Compute inverse
            inversed = tuple(tuple(c[::-1]) for c in self.cycles[::-1])
            self.inv = permutation(*inversed)
        return self.inv

    def acts_on(self):
        return {x for c in self.cycles for x in c}

    def canonicalize(self):
        if len(self.cycles) == 1:
            to_visit = sorted(self.acts_on())
            if len(to_visit) < 2:
                self.cycles = []
                return
            cycle = tuple(i for i in find_cycle(self, to_visit[0]))
            self.cycles = [cycle]
            return

        to_visit = self.acts_on()
        hopper = sorted(to_visit)
        cycles = []
        for x in hopper:
            if x in to_visit:
                cycle = tuple(i for i in find_cycle(self, x))
                if len(cycle) > 1:
                    cycles.append(Cycle(cycle))
                to_visit.difference_update(cycle)
        self.cycles = cycles

    def conjugate(self, other):
        inv = other.inverse()
        return inv * self * other

    def support(self):
        s = set()
        for c in self.cycles:
            s.update(c)
        return s

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
        p1 = self.cycles
        p2 = other.cycles
        return permutation(*(self.cycles + other.cycles))

    def __repr__(self):
        return "<Permutation {}>".format(str(self))

    def __str__(self):
        return ''.join([str(c) for c in self.cycles])

    def __pow__(self, power, modulo=None):
        if not isinstance(power, int) and not isinstance(power, np.integer):
            raise RuntimeError("Cannot take non-integer power of a Permutation")
        # TODO: Square Multiply
        if power == 0:
            return Cycle(())
        p = power if power > 0 else -power
        result = self
        for i in range(1, p):
            result = result * self

        return result if power > 0 else result.inverse()

    def __eq__(self, other):
        # TODO: make more robust
        return sorted(self.cycles) == sorted(other.cycles)


class Cycle(Permutation):
    def __init__(self, cycle):
        assert isinstance(cycle, tuple) or isinstance(cycle, list)
        if len(set(cycle)) != len(cycle):
            raise RuntimeError("Cycle {} has duplicate entry".format(cycle))
        super().__init__([cycle])
        self.cycle = cycle

    def inverse(self):
        """
        Compute the inverse of this cycle
        :return: inverse of self
        """
        if self.inv is None:
            self.inv = Cycle(self.cycle[::-1])
            self.inv.inverse = self

        return self.inverse

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

    def __iter__(self):
        return iter(self.cycle)

    def __repr__(self):
        return "<Cycle {}>".format(str(self))

    def __str__(self):
        return str(self.cycle)