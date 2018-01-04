class DotGenerator:

    @staticmethod
    def generate(graph, labels=None, size=(1, 1)):
        if graph.directed():
            return DotGenerator.generate_digraph(graph, size)
        else:
            return DotGenerator.generate_graph(graph, size)

    @staticmethod
    def generate_digraph(graph, labels=None, size=(1, 1)):
        pass

    @staticmethod
    def generate_graph(graph, labels=None, size=(1, 1)):
        header = ''' graph g {'''
        footer = '''}'''
        pass
