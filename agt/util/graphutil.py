from agt.graph import MatrixGraph
from agt.math.common import choose
from agt.util.arrayops import ones
from itertools import combinations

class GraphUtil:
    @staticmethod
    def complete(order, type='matrix'):
        if type == 'matrix':
            return MatrixGraph(order=order, edges=ones((order, order)))
        else:
            raise RuntimeError("Unrecognized Graph type: {}".format(type))

    @staticmethod
    def random(order, density=0.5, type='matrix'):
        if type == 'matrix':
            return MatrixGraph.create_random(order, density)
        else:
            raise RuntimeError("Unrecognized Graph type: {}".format(type))

    @staticmethod
    def C(n, circulant_set=None):
        if circulant_set is None:
            circulant_set = {1}
        cs = {int(x) for x in circulant_set}.union({-int(x) for x in circulant_set})

        def edge_predicate(i,j):
            x,y = int(i), int(j)    # In case we are using uint32s
            return (x - y) % n in cs or (y - x) % n in cs

        g = MatrixGraph(order=n, edges=edge_predicate)
        return g

    @staticmethod
    def J(v, k, i):
        """
        Create a Johnson graph over a set of size v with samples of size k and
        intersection size i.

        :param v:
        :param k:
        :param i:
        :return:
        """
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

    @staticmethod
    def K(n, m=None):
        if m is None:
            # Return complete order-n graph
            def edge_predicate(i, j):
                return True
            return MatrixGraph(order=n, edges=edge_predicate)

        else:
            # Complete bipartite graph of orders m, n
            def edge_predicate(i, j):
                return i < n <= j or j < n <= i

            return MatrixGraph(order=n + m, edges=edge_predicate)

