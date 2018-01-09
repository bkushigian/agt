"""
Graphs
"""
import numpy as np
import random
from collections import Iterable
from itertools import product
from agt.util.arrayops import generate_lower_triangle, dense_adjacency_matrix
from sympy.combinatorics.permutations import Permutation

# These are constants so we declare them once to make life easy
from agt.util.dot import DotGenerator

one = np.uint32(1)
zero = np.uint32(0)


def extract_edge(a, b):
    """
    A convenience function that takes a,b where:
        * a is a tuple representing an edge and b is none, or
        * a, b are both ints representing the start/end of the edge
    """
    if b is None:
        assert (isinstance(a, tuple) or isinstance(a, set) or isinstance(a, Edge)) and (len(a) == 2)
        a, b = a
    return Edge(a, b)


class Graph:
    def E(self, X=None, Y=None):
        """
        E(u,v) defines three different classes of edges;
        (1) E(u=X,v=Y): If X and Y are subsets of V, E(X,Y) returns all edges
            e = xy where x is in X and y is in Y.
        (2) E(X): If X is a subset of V, then E(X) returns all edges e = xy
            where x is in X; this is equivalent to E(X, V)
        (3) E(): Return all edges. This is equivalent to E(V,V) E() returns the
            edge set of this graph

        As syntactic sugar, for x in V we write E(x) to denote E({x}) (and
        similarly for y values)
        :param X: Subset of vertices
        :param Y: Subset of vertices
        :return: All edges xy with X and Y as ends
        """
        raise NotImplementedError()

    def V(self):
        """Returns the vertex set of this graph"""
        raise NotImplementedError()

    def is_edge(self, a, b=None):
        """Predicate that tests if {a,b} is in our edge set"""
        raise NotImplementedError()

    def add(self, a, b=None, duplicate=False):
        """Add {a,b} to our edge set"""
        raise NotImplementedError()

    def adjacent(self, v):
        """Get a list of all nodes adjacent to node v"""
        raise NotImplementedError()

    def as_dot(self, labels=None):
        """Represent this graph as a Dot file"""
        raise NotImplementedError()

    def complement(self):
        """Calculate the complement graph of self"""
        raise NotImplementedError()

    def connected(self):
        """Predicate that computes if self is connected"""
        raise NotImplementedError()

    def density(self):
        """Calculate the density of this graph"""
        raise NotImplementedError()

    def difference(self, other):
        """Return the difference of this graph and other, where this is defined
        on graphs of equal order n to be the set of edges in self not in other."""
        raise NotImplementedError()

    def directed(self):
        """Return true if this is a directed graph, false if it is an undirected graph"""
        raise NotImplementedError()

    def distance(self, a, b):
        """Return the distance between vertices a and b. If unconnected, return -1"""
        raise NotImplementedError()

    def edges(self):
        """Return a generator over the edges"""
        raise NotImplementedError()

    def intersection(self, other):
        """If self = (V,E) and other = (V', E'), return (V ∩ V', E ∩ E')"""
        raise NotImplementedError()

    def is_fixed_by(self, p):
        """
        Test if this graph is fixed by a permutation p
        :param p: a Permutation to be tested to see if its in Aut(self)
        :return: True if `p` is in Aut(self), False otherwise
        """
        raise NotImplementedError()

    def is_subgraph(self, other):
        raise NotImplementedError()

    def is_supergraph(self, other):
        return other.is_subgraph(self)

    def nodes(self):
        """Return the set of nodes"""
        raise NotImplementedError()

    def order(self):
        """Return the number of vertices in self"""
        raise NotImplementedError()

    def permute(self, p):
        """
        Permute the vertices of the graph by permutation `p`, creating a new
        graph.
        :param p:
        :return:
        """
        raise NotImplementedError()

    def regularity(self):
        """
        A graph is k-regular if each vertex has valency k. Graph.regularity()
        calculates if such a k exists and, if so, returns that k; otherwise
        it returns None
        :return: k if G is k-regular, None otherwise
        """
        raise NotImplementedError()

    def remove(self, a, b=None, duplicate=False):
        """Remove an edge"""
        raise NotImplementedError()

    def size(self):
        """Return the number of edges in self"""
        raise NotImplementedError()

    def union(self, other):
        """If self = (V, E) and other = (V', E'), return (V ∪ V', E ∪ E')"""
        raise NotImplementedError()

    def valency(self, v):
        """
        The valency of a vertex is the number of vertices adjacent to it.
        :param v: an element of the vertex set
        :return: the number of adjacent vertices to `v`
        """
        if v not in self.__v:
            raise ValueError("{} is not a vertex".format(v))
        return len(self.adjacent(v))

    def vertices(self):
        """
        Vertices returns a copy of the vertex set
        :return: a copy of the vertex set
        """

    def symmetric_difference(self, other):
        """Defined to be the union of self - other UNION other - self"""
        return self.difference(other).union(other.difference(self))

    def __eq__(self, other):
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
                for _ in self.edges():
                    self.__size += 1

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

                    elif isinstance(e0, Edge):
                        order = 0
                        for a, b in edges:
                            order = max(order, max(a, b))
                        order += 1

                        self.__order = order
                        self.__v = np.arange(order)
                        self.__e = np.zeros((self.order(), self.order()), np.uint32)

                        for e in edges:
                            self.add(e)

                    elif isinstance(e0, list):  # Adjacency list
                        order = len(edges)
                        self.__order = order
                        self.__v = np.arange(order)
                        self.__e = np.zeros((self.__order, self.__order), np.uint32)
                        for i, subl in enumerate(edges):
                            for j in subl:
                                self.add(i, j)

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
                for _ in self.edges():
                    self.__size += 1

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

            elif callable(edges):
                self.__e = np.zeros((order, order), np.uint32)
                for i, j in product(self.__v, self.__v):
                    self.__e[i][j] = one if (i != j) and edges(i, j) else zero
            else:
                self.__e = np.zeros((self.order(), self.order()), np.uint32)

    def E(self, X=None, Y=None):
        if X is None and Y is None:
            return self.edges()
        if X is None:
            X = self.V()
        elif Y is None:
            Y = self.V()

        if isinstance(X, int) or isinstance(X, np.integer):
            X = {X}

        if isinstance(Y, int) or isinstance(Y, np.integer):
            Y = {Y}

        result = {Edge(x, y) for x in X for y in Y if (x, y) in self}
        return result

    def V(self):
        return self.__v

    def add(self, a, b=None, duplicate=False):
        if duplicate:
            return MatrixGraph(order=self.order(), edges=self.__e.copy()).add(a,b)

        a, b = extract_edge(a, b)
        if 0 <= a < self.order() and 0 <= b < self.order():
            if not self.is_edge(a, b):
                self.__e[a][b] = one
                self.__e[b][a] = one
                self.__size += 1
        else:
            raise IndexError("Cannot add edge ({},{}) to graph of order {}".format(a,b,self.order()))
        return self

    def adjacent(self, v):
        assert 0 <= v < self.order()
        return {i for i, u in enumerate(self.__e[v]) if u}

    def as_dot(self, labels=None, prefix='n'):
        if not labels:
            labels = []

        if len(labels) < self.order():
            # Pad our labels to include a label for each vertex
            labels += list(map(lambda n: "{}{}".format(prefix, n), range(len(labels), self.order())))

        dot = DotGenerator.generate(self, labels)

        return dot

    def complement(self):
        edges = dense_adjacency_matrix(self.__e.shape[0]) - self.__e
        g = MatrixGraph(order=self.order(), edges=edges)
        return g

    def connected(self):
        visited = set()
        hopper = set([0])
        while hopper:
            new_hopper = set()
            for x in hopper:
                if x not in visited:
                    visited.add(x)
                    new_hopper.update(self.adjacent(x))
            hopper = new_hopper.difference(visited)
        for v in self.__v:
            if v not in visited:
                return False
        return True

    @staticmethod
    def create_random(order, density=0.5):
        g = MatrixGraph(order=order)
        for (a, b) in generate_lower_triangle(order):
            if random.random() < density:
                g.add(a, b)
        return g

    def density(self):
        return 2 * self.size() / (self.order() * (self.order() - 1))

    def difference(self, other):
        assert self.order() == other.order()

        def edge_predicate(i, j):
            if i == j:
                return False
            return (i, j) in self and (i, j) not in other

        return MatrixGraph(order=self.order(), edges=edge_predicate)

    def directed(self):
        return False

    def distance(self, a, b):
        assert 0 <= a < self.order() and 0 <= b < self.order()
        d = 0
        visited = set()
        boundary = set([a])
        new_boundary = set()
        while boundary:
            for x in boundary:
                if x not in visited:
                    if x == b:
                        return d
                    new_boundary.update(self.adjacent(x))
                    visited.add(x)
            d += 1
            boundary = new_boundary.difference(visited)
            new_boundary = set()

        return -1

    def edges(self):
        return {Edge(i, j) for i in self.__v for j in self.__v if j > i and self.is_edge(i, j)}

    def intersection(self, other):
        order = min(self.order(), other.order())

        def edge_predicate(i, j):
            if i == j:
                return False
            return (i, j) in self and (i, j) in other

        return MatrixGraph(order=order, edges=edge_predicate)

    def is_edge(self, a, b=None):
        if 0 <= a < self.order() and 0 <= b < self.order():
            return self.__e[a][b] == one
        return False

    def is_fixed_by(self, p):
        for a, b in self.edges():
            if (p(a), p(b)) not in self:
                return False

    def is_subgraph(self, other, strict=True):
        if strict:
            if self.order() != other.order():
                return False
        elif self.order() > other.order():
            return False
        for e in self.edges():
            if e not in other:
                return False
        return True

    def matrix(self):
        return self.__e.copy()

    def nodes(self):
        return self.__v

    def permute(self, p):
        assert isinstance(p, Permutation)
        # make sure that p is the right size
        order = max(self.order(), p.size)
        if p.size < order:
            p = Permutation(p, size=order)
        edges = np.zeros((order, order), dtype=np.uint32)
        for i in range(order):
            for j in range(order):
                if max(i, j) < self.order():
                    edges[i][j] = self.__e[p(i)][p(j)]
        return MatrixGraph(edges=edges)

    def order(self):
        return self.__order

    def remove(self, a, b=None, duplicate=False):
        if duplicate:
            return MatrixGraph(order=self.order(), edges=self.__e.copy()).remove(a,b)

        a, b = extract_edge(a, b)
        if 0 <= a < self.order() and 0 <= b < self.order():
            if self.is_edge(a, b) or self.is_edge(b, a):
                self.__e[a][b] = zero
                self.__e[b][a] = zero
                self.__size -= 1
        else:
            raise IndexError("Cannot remove edge ({},{}) to graph of order {}".format(a,b,self.order()))
        return self

    def size(self):
        return self.__size

    def union(self, other):
        order = max(self.order(), other.order())

        def edge_predicate(i, j):
            if i == j:
                return False
            return (i, j) in self or (i, j) in other

        return MatrixGraph(order=order, edges=edge_predicate)

    def vertices(self):
        return self.__v.copy()

    def view(self, dest=None, size=(7, 5), labels=None):
        DotGenerator.compile_dot(self, dest=dest, size=size, labels=labels, view=True)

    def write_dot(self, dest=None, size=(7, 5), labels=None):
        DotGenerator.write_dot(self, labels=labels, size=size, dest=dest)

    def xor(self, other):
        assert self.order() == other.order(), "Cannot take the xor of different ordered graphs"

        def edge_predicate(i, j):
            if i == j:
                return False
            return (((i, j) in self) + ((i, j) in other)) % 2

        return MatrixGraph(order=self.order(), edges=edge_predicate)

    def __contains__(self, item):
        if isinstance(item, Iterable) and len(item) is 2:
            a, b = item
            return self.is_edge(a, b)
        return False

    def __eq__(self, other):
        if self.size() != other.size():
            return False

        if self.order() != other.order():
            return False

        my_edges = self.edges()
        their_edges = other.edges()

        return my_edges == their_edges

    def __iter__(self):
        """Iterate through edges"""
        for a in self.__v:
            for b in self.__v:
                if b > a:
                    break
                if self.is_edge(a, b):
                    yield Edge(a, b)


# XXX: This is not ready for use - DO NOT USE
class Edge:
    def __init__(self, x, y, weight=None):
        self.x = x
        self.y = y
        self.weight = weight

    def __hash__(self):
        return hash(frozenset((self.y, self.x)))

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        """This is simply for convenience."""
        return 2

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return {self.x, self.y} == {other.x, other.y}

    def __str__(self):
        return '{' + '{},{}'.format(self.x, self.y) + '}'

    def __repr__(self):
        return '{' + '{},{}'.format(self.x, self.y) + '}'


class EdgeSet:
    def __init__(self):
        pass
