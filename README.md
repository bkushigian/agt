AGT
===

`agt` is a basic algebraic graph theory library. It provides a simple API to
play around with/implement different graph-theoretic algorithms.

## Dependencies

The `agt` project is built using Python3.5 and depends on

* NumPy
* SymPy

Both of these can be installed using `pip install numpy` and `pip install sympy`

## Using AGT

Note that this is under development and is subject to change. Here are the
basics of using AGT.

`agt` is simply to use - just import it. 

Here is a sample session:

    >>> from agt.graph import MatrixGraph
    >>> G = MatrixGraph.create_random(order=32, density=0.75)
    >>> G.size()    # Number of edges
    21
    >>> G.order()   # Number of vertices
    10
    >>> G.E()       # Get a list of edges
    {{5,7}, {4,8}, {1,9}, {1,2}, {2,6}, {1,4}, {3,5}, {6,7}, {2,7}, {8,9}, {3,4}, {5,9}, {2,3}, {0,2}, {1,6}, {2,5}, {7,8}, {6,8}, {7,9}, {0,3}, {0,1}}
    >>> G.E(0)      # Get a list of edges incident with vertex 0
    {{0,2}, {0,3}, {0,1}}
    >>> G.E({0,1,2,3,4}, {5,6,7,8,9})   # Get edges connecting sets  {0, 1, 2, 3, 4} and {5, 6, 7, 8, 9}
    {{4,8}, {1,9}, {2,6}, {3,5}, {2,5}, {1,6}, {2,7}}
    >>> G.density()
    0.4666666666666667

We can also specify the edge set at construction time.

    >>> H = MatrixGraph(order=4, edges=[(0,1), (1,2), (2,3), (3,0)])
    >>> H.order()
    4
    >>> H.size()
    4
    >>> (0,1) in H
    True
    >>> (1,0) in H
    True
    >>> H.matrix()
    [[0, 1, 0, 1],
     [1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 0, 1, 0]]
    >>> c = H.complement()
    >>> c.matrix()
    [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, 1, 0, 0]]
    >>> H.density()      # Calculate the density of the edge set
    0.666666666666666
    >>> (0,2) in H
    False
    >>> H.add(0,2).add((1,3))
    <agt.graph.MatrixGraph object at 0x7f1c2a3a6fd8>
    >>> (0,2) in H
    True
    >>> H.density()
    1.0

Basic graph-theoretic operations are supported:

    >>> G = MatrixGraph(order=4).add(0,1).add(1,2).add(2,3)
    >>> K = G.union(G.complement())          # The complete graph (K_4)
    >>> print(K.matrix())
    [[0 1 1 1]
     [1 0 1 1]
     [1 1 0 1]
     [1 1 1 0]]
    >>> E = G.intersection(G.complement())   # The empty graph
    >>> print(E.matrix())
    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]

For more information, browse the Graph API located in `agt/graphs.py`

## Development and Contributions

Feel free to fork and tinker. Issue pull requests to contribute.
