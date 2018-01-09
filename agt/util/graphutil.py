from agt.graph import MatrixGraph
from agt.math.common import choose
from agt.util.arrayops import ones
from itertools import combinations
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup


def automorphism_group(graph):
    """
    This is wildly inefficient!
    :param graph:
    :return:
    """
    order = graph.order()
    grp = {Permutation(order)}          # The identity group
    for g in SymmetricGroup(order).generate():
        if graph.is_fixed_by(g):
            grp.add(g)
    return PermutationGroup(*grp)


def bipartite(m, n, type='matrix'):
    """
    Create a bipartite graph on (m,n) veritces
    :param m:
    :param n:
    :param type:
    :return:
    """

    if type == 'matrix':
        def edge_predicate(i, j):
            return i < m == j >= m
        return MatrixGraph(order=m+n, edges=edge_predicate)
    else:
        raise RuntimeError("Unrecognized Graph type: {}".format(type))


def complete(order, type='matrix'):
    if type == 'matrix':
        return MatrixGraph(order=order, edges=ones((order, order)))
    else:
        raise RuntimeError("Unrecognized Graph type: {}".format(type))


def empty(order, type='matrix'):
    if type == 'matrix':
        return MatrixGraph(order, edges=lambda x, y: False)
    else:
        raise RuntimeError("Unrecognized Graph type: {}".format(type))


def random(order, density=0.5, type='matrix'):
    if type == 'matrix':
        return MatrixGraph.create_random(order, density)
    else:
        raise RuntimeError("Unrecognized Graph type: {}".format(type))


def C(n, circulant_set=None, type='matrix'):

    if type == 'matrix':
        if circulant_set is None:
            circulant_set = {1}
        cs = {int(x) for x in circulant_set}.union({-int(x) for x in circulant_set})

        def edge_predicate(i,j):
            x, y = int(i), int(j)    # In case we are using uint32s
            return (x - y) % n in cs or (y - x) % n in cs

        g = MatrixGraph(order=n, edges=edge_predicate)
        return g
    else:
        raise RuntimeError("Unrecognized Graph type: {}".format(type))


def J(v, k, i, type='matrix'):
    """
    Create a Johnson graph over a set of size v with samples of size k and
    intersection size i.

    :param v:
    :param k:
    :param i:
    :return:
    """
    if type == 'matrix':
        xs = [x for x in range(v)]
        order = choose(v,k)
        vertices = []    # A list of length order with combinations below
        for a, c in enumerate(combinations(xs, k)):
            print(a)
            vertices.append(c)

        def edge_predicate(x,y):
            if len(set(vertices[x]).intersection(set(vertices[y]))) == i:
                return True
            return False
        return MatrixGraph(order=order, edges=edge_predicate)
    else:
        raise RuntimeError("Unrecognized Graph type: {}".format(type))


def K(n, m=None, type='matrix'):
    if type == 'matrix':
        if m is None:
            return complete(order=n)

        else:
            return bipartite(m, n)
    else:
        raise RuntimeError("Unrecognized Graph type: {}".format(type))


def E(graph):
    return graph.edges()


def V(graph):
    return graph.vertices()
