from datetime import datetime
from os import path as osp
from tempfile import mkstemp, mkdtemp
import graphviz


tempdir = mkdtemp(prefix='AGT_')
print('tempdir:', tempdir)


class DotGenerator:
    @staticmethod
    def generate(graph, labels=None, size=(7, 5)):
        if graph.directed():
            return DotGenerator.generate_digraph(graph, labels, size)
        else:
            return DotGenerator.generate_graph(graph, labels, size)

    @staticmethod
    def generate_digraph(graph, labels=None, size=(7, 5)):
        pass

    @staticmethod
    def generate_graph(graph, labels=None, size=(7.5, 10), comment=""):
        if labels is None:
            labels = []

        if len(labels) < graph.order():
            labels += ['n{}'.format(i) for i in range(len(labels), graph.order())]

        dot = graphviz.Graph(comment=comment)
        for node in graph.V():
            dot.node(str(node))
        for edge in graph.E():
            dot.edge(str(list(edge)[0]), str(list(edge)[1]))
        return dot

    @staticmethod
    def compile_dot(graph, labels=None, size=(7.5, 10), dest=None, view=False):
        if dest is None:
            _, dest = mkstemp(dir=tempdir, suffix='.dot')
        dot = DotGenerator.generate(graph, labels=labels, size=size)
        res = dot.render(dest, view=view)
        return dest, res



