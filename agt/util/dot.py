from datetime import datetime


class DotGenerator:
    @staticmethod
    def generate(graph, labels=None, size=(1, 1)):
        if graph.directed():
            return DotGenerator.generate_digraph(graph, labels, size)
        else:
            return DotGenerator.generate_graph(graph, labels, size)

    @staticmethod
    def generate_digraph(graph, labels=None, size=(1, 1)):
        pass

    @staticmethod
    def generate_graph(graph, labels=None, size=(7.5, 10)):
        if labels is None:
            labels = []

        if len(labels) < graph.order():
            labels += ['n{}'.format(i) for i in range(len(labels), graph.order())]

        header = ''' graph g {'''
        size = '''size=\"{},{}\"'''.format(size[0], size[1])
        nodes = '\n    '.join(map(str, labels))
        edges = '\n    '.join(['{} -- {}'.format(labels[i], labels[j]) for (i, j) in iter(graph)])
        footer = '''}\n'''
        return '\n    '.join([header, size, edges, footer])

    @staticmethod
    def write_dot(graph, labels=None, size=(7.5, 10), dest=None):
        if dest is None:
            time = str(datetime.now()).split('.')[0]
            dest = 'graph{}.dot'.format(time)

        print("Writing DOT file to {}".format(dest))

        with open(dest, 'w') as f:
            f.write(DotGenerator.generate(graph, labels, size))
