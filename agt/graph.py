"""
Graphs
"""
import numpy as np
import random
from collections import Iterable
from agt.util.arrayops import generate_lower_triangle, ones

# These are constants so we declare them once to make life easy
one = np.uint32(1)
zero = np.uint32(0)


def extract_edge(a, b):
    """
    A convenience function that takes a,b where:
        * a is a tuple representing an edge and b is none, or
        * a, b are both ints representing the start/end of the edge
    """
    if b is None:
        assert (isinstance(a, tuple) or isinstance(a, set)) and (len(a) == 2)
        a, b = tuple(a)
    return {a, b}


class Graph:
    def E(self, a, b):
        """Predicate that tests if {a,b} is in our edge set"""
        raise NotImplementedError()

    def add(self, a, b=None):
        """Add {a,b} to our edge set"""
        raise NotImplementedError()

    def as_dot(self, labels=None):
        """Represent this graph as a Dot file"""
        raise NotImplementedError()

    def directed(self):
        """Return true if this is a directed graph, false if it is an undirected graph"""
        raise NotImplementedError()

    def edges(self):
        """Return a generator over the edges"""
        raise NotImplementedError()

    def remove(self, a, b=None):
        """Remove an edge"""
        raise NotImplementedError()

    def complement(self):
        """Calculate the complement graph of self"""
        raise NotImplementedError()

    def order(self):
        """Return the number of vertices in self"""
        raise NotImplementedError()

    def size(self):
        """Return the number of edges in self"""
        raise NotImplementedError()

    @staticmethod
    def create_random(order, density=0.5):
        """Create a random graph of a given order and edge density"""
        raise NotImplementedError()


class MatrixGraph(Graph):
    """
    Adjacency matrix representation of a graph
    """

    def __init__(self, order=None, edges=None):
        self.__size = 0
        if order is None:
            if isinstance(edges, np.ndarray):   # This is an adjacency matrix
                self.__order = len(edges)
                self.__v = set(np.arange(self.__order, dtype=np.uint32))
                self.__e = edges.copy()

            elif isinstance(edges, list):       # This is an adjacency list
                if len(edges) is 0:
                    raise RuntimeError("No order or edges specified")
                else:
                    e0 = edges[0]
                    if (isinstance(e0, tuple) or isinstance(e0, set) or isinstance(e0, frozenset)) and len(e0) == 2:
                        # Here, since order isn't specified, we take it to be the maximum
                        # value in any of the tuples
                        order = 0
                        for a, b in edges:
                            order = max(order, max(a, b))
                        order += 1   # We are zero indexing, so account for offset

                        self.__order = order
                        self.__v = np.arange(order)
                        self.__e = np.zeros((self.order(), self.order()), np.uint32)

                        for e in edges:
                            self.add(e)

                    elif isinstance(edges[0], list):  # Adjacency list
                        order = len(edges)
                        self.__order = order
                        self.__v = np.arange(order)
                        self.__e = np.zeros((self.__order, self.__order), np.uint32)
                        for i, subl in enumerate(edges):
                            for j in subl:
                                self.add(i,j)

            elif edges:
                raise RuntimeError("Unrecognized edge input: {}".format(edges))

            else:
                raise RuntimeError("Must specify edges or order")

            # Calculate size from matrix

        else:
            assert order > 0
            self.__order = order
            self.__v = set(np.arange(order, dtype=np.uint32))

            if isinstance(edges, np.ndarray):   # This is an adjacency matrix
                if order != len(edges):
                    raise RuntimeError("In MatrixGraph order of {} specified but edges has shape {}".format(order, edges.shape))
                self.__e = edges.copy()

            elif isinstance(edges, list):       # This is an adjacency list
                self.__e = np.zeros((self.__order, self.__order), np.uint32)
                if len(edges) > 0:
                    if isinstance(edges[0], tuple):  # List of tuples
                        # Here, since order isn't specified, we take it to be the maximum
                        # value in any of the tuples
                        for a, b in edges:
                            self.add(a, b)

                    elif isinstance(edges[0], list):  # Adjacency list
                        for i, subl in enumerate(edges):
                            for j in subl:
                                self.add(i, j)

            else:
                self.__e = np.zeros((self.order(), self.order()), np.uint32)

    def add(self, a, b=None):
        a, b = extract_edge(a, b)
        if 0 <= a < self.order() and 0 <= b < self.order():
            if not self.E(a, b):
                self.__e[a][b] = one
                self.__e[b][a] = one
                self.__size += 1
        else:
            raise IndexError("Cannot add edge ({},{}) to graph of order {}".format(a,b,self.order()))
        return self

    def as_dot(self, labels=None):
        if not labels:
            labels = []

        if len(labels) < self.order():
            # Pad our labels to include a label for each vertex
            labels += list(map(lambda n : "n{}".format(n), range(len(labels), self.order())))

        raise NotImplementedError()

    def complement(self):
        edges = ones(self.__e.shape) - self.__e
        g = MatrixGraph(edges=edges)
        return g

    @staticmethod
    def create_random(order, density=0.5):
        g = MatrixGraph(order=order)
        for (a, b) in generate_lower_triangle(order):
            if random.random() < density:
                g.add(a, b)
        return g

    def directed(self):
        return False

    def E(self, a, b=None):
        return self.__e[a][b] == one

    def edges(self):
        return {frozenset((i, j)) for i in self.__v for j in self.__v if j > i and self.E(i, j)}

    def order(self):
        return self.__order

    def remove(self, a, b=None):
        a, b = extract_edge(a, b)
        if 0 <= a < self.order() and 0 <= b < self.order():
            if self.E(a, b) or self.E(b, a):
                self.__e[a][b] = zero
                self.__e[b][a] = zero
                self.__size -= 1
        else:
            raise IndexError("Cannot remove edge ({},{}) to graph of order {}".format(a,b,self.order()))
        return self

    def size(self):
        return self.__size

    def __contains__(self, item):
        if isinstance(item, Iterable) and len(item) is 2:
            a, b = item
            return self.E(a, b)
        return False

    def __iter__(self):
        """Iterate through edges"""
        for a in self.__v:
            for b in self.__v:
                if b > a:
                    break
                if self.E(a, b):
                    yield {a, b}

