"""
Graphs
"""
import numpy as np
import random
from math import inf
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

    def d(self, v):
        """
        Return the degree (or valency) of a vertex v.

        Example:
            >>> g = MatrixGraph(order=4).add(0,1).add(1,2).add(2,3).add(3,0)
            >>> g.d(0)
            2
            >>> g.add(0,2)
            >>> g.d(0)
            3
        :param v: vertex in self.V
        :return: degree of v
        """
        raise NotImplementedError()

    def density(self):
        """
        Return the percentage of possible edges that are present. This is
        implemented differently for directed/undirected graphs.

        Example of Undirected Graph:
            >>> g = Graph(order=5).add(0,1).add(1,2).add(2,3).add(3,4).add(4,0)
            >>> g.density()
            0.5
            >>> g.size() / (g.size() + g.complement().size())
            0.5
        """
        raise NotImplementedError()

    def difference(self, other):
        """
        Return the difference of this graph and other, where this is defined
        on graphs of equal order n to be the set of edges in self not in other.

        Example:
            >>> K4 = Graph(order=4, edges=[(0,1), (1,2), (2,3), (3,0), (0,2), (1,3)]
            >>> C4 = Graph(order=4, edges=[(0,1), (1,2), (2,3), (3,0)])
            >>> K4.difference(C4).E()    # Get the edges of the difference
            {{1,3}, {0,2}}
        """
        raise NotImplementedError()

    def directed(self):
        """Return true if this is a directed graph, false if it is an undirected graph"""
        raise NotImplementedError()

    def distance(self, a, b):
        """
        Return the distance between vertices a and b. If unconnected, return math.inf.

        Example:
            >>> g = Graph(order=5).add(0,1).add(1,2).add(2,3).add(3,4)
            >>> g.distance(0,4)
            4
            >>> g = Graph(order=5).add(0,1).add(2,3)   # Disconnected
            >>> g.distance(0,3)
            math.inf
        """
        raise NotImplementedError()

    def intersection(self, other):
        """
        If self = (V,E) and other = (V', E'), return (V ∩ V', E ∩ E')

        Example:
            >>> G = Graph(order=10).add(0,1).add(2,3).add(4,5).add(8,9)
            >>> H = Graph(order=5).add(0,1).add(1,2).add(2,3)
            >>> GnH = G.intersection(H)
            >>> GnH.order()
            5
            >>> GnH.V()
            {0,1,2,3,4}
            >>> GnH.E()
            {{0,1}, {2,3}}
        """
        raise NotImplementedError()

    def is_fixed_by(self, p):
        """
        Test if this graph is fixed by a permutation p

        Example:
            >>> from sympy.combinatorics.permutations import Permutation
            >>> G = Graph(order=4).add(0,1).add(1,2).add(2,3).add(3,0)
            >>> p = Perm(0,1,2,3)
            >>> G.is_fixed_by(p)
            True
            >>> q = Perm(0,1,2)
            >>> G.is_fixed_by(q)
            False
        :param p: a Permutation to be tested to see if its in Aut(self)
        :return: True if `p` is in Aut(self), False otherwise
        """
        raise NotImplementedError()

    def is_subgraph(self, other):
        raise NotImplementedError()

    def is_supergraph(self, other):
        return other.is_subgraph(self)

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
                for _ in self.E():
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
                for _ in self.E():
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
                    if i != j and edges(i, j):
                        self.add(i, j)
            else:
                self.__e = np.zeros((self.order(), self.order()), np.uint32)

    def E(self, X=None, Y=None):
        if X is None and Y is None:
            return {Edge(i, j) for i in self.__v for j in self.__v if j > i and self.is_edge(i, j)}
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

    def d(self, v):
        if v not in self.V():
            raise ValueError("Unexpected value {}: expected a vertex".format(v))
        return len(self.E(v))

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

        return inf

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
        for a, b in self.E():
            x = p(a) if a < p.size else a
            y = p(b) if b < p.size else b
            if Edge(x, y) not in self:
                return False
        return True

    def is_subgraph(self, other, strict=True):
        if strict:
            if self.order() != other.order():
                return False
        elif self.order() > other.order():
            return False
        for e in self.E():
            if e not in other:
                return False
        return True

    def matrix(self):
        return self.__e.copy()

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

        my_edges = self.E()
        their_edges = other.E()

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
