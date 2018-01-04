from agt.graph import MatrixGraph
from agt.util.arrayops import ones

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

