"""
Miscellaneous functions to play with
"""

from itertools import product, combinations
from agt.graph import MatrixGraph


def count_char(s, c='1'):
    """
    Count the number of occurences of c in s
    :param s: string to count occurrences in
    :param c: character to count
    :return: number of times c occurs in s
    """
    return sum(x == c for x in s)


def do_you_speak_barrington(g_string):
    """
    Given a representation of a graph of the form:
        0 01 101 1111
    create the associated graph.

    :param g_string: string containing the graph information
    :return: The appropriate MatrixGraph
    """
    if ':' in g_string:
        g_string = g_string[g_string.index(':'):]
    g_string = g_string.strip()
    g_strings = g_string.split()
    order = len(g_strings) + 1
    G = MatrixGraph(order=order)
    for i, s in enumerate(g_strings):
        for j, c in enumerate(s):
            if c == '1':
                G.add(i+1, j)
    return G


def sample_no_replacement(collection, min=1, max=-1):
    if max == -1:
        max = len(collection)

    result = set()
    for r in range(min, max + 1):
        result.update(combinations(collection, r))
    return result


def is_clique(c, G):
    for (a,b) in product(c, repeat=2):
        if a != b and not G.is_edge(a,b):
            return False
    return True


def list_cliques(G):
    V = G.V()
    E = G.E()
    cliques = set()


def is_line_graph(G):
    # TODO: Write test for a line graph
    pass
