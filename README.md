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

`agt` is simply to use - just import it. 

Here is a sample session:

    >>> from agt.graph import MatrixGraph
    >>> g = MatrixGraph.create_random(order=32, density=0.75)
    >>> g.size()    # Number of edges
    376
    >>> g.order()   # Number of vertices
    32
    >>> g.edges()   # Get a numpy adjacency matrix
	[[0, 1, 1, ..., 1, 1, 1],
	 [1, 0, 1, ..., 0, 1, 0],
	 [1, 1, 0, ..., 1, 1, 1],
	 ..., 
	 [1, 0, 1, ..., 0, 1, 0],
	 [1, 1, 1, ..., 1, 0, 1],
	 [1, 0, 1, ..., 0, 1, 0]]
    >>> g = MatrixGraph(order=4, edges=[(0,1), (1,2), (2,3), (3,0)])
    >>> g.order()
    4
    >>> g.size()
    4
    >>> (0,1) in g
    True
    >>> (1,0) in g
    True
    >>> g.matrix()
    [[0, 1, 0, 1],
     [1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 0, 1, 0]]
    >>> c = g.complement()
    >>> c.matrix()
    [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, 1, 0, 0]]
    >>> g.density()      # Calculate the density of the edge set
    0.666666666666666
    >>> (0,2) in g
    False
    >>> g.add(0,2).add((1,3))
    <agt.graph.MatrixGraph object at 0x7f1c2a3a6fd8>
    >>> (0,2) in g
    True
    >>> g.density()
    1.0

## Development and Contributions

Feel free to fork and tinker. Issue pull requests to contribute
