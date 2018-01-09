"""
arrayops stores various array operations that will come in handy
"""

import bisect
import numpy as np

zero = np.uint32(0)
one = np.uint32(1)


def index(a, x):
    """
    Given a sorted list, locate the leftmost value equal to x and return its
    index. This is the same as a.index(x) but uses the additional information
    that a is a sorted list.

    Example:
        >>> a = [0,1,2,3,4]
        >>> ind = index(a, 3)
        >>> ind
        3

    :param a: List to search
    :param x: value to find
    :return:
    """
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise None


def ones(shape, dtype=np.uint32):
    """
    Create an numpy array filled with the value one

    Example:

        >>> ones((3,3))
        np.ndarray([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]], dtype=np.uint32)
    :param shape:
    :param dtype:
    :return:
    """
    return np.full(shape, one, dtype=dtype)


def diagonal(n, dtype=np.uint32):
    """
    Create a matrix with one along the diagonal and zeros everywhere else.

    Example:

        >>> diagonal(3)
        np.ndarray([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=np.uint32)

    :param n: height/width of array
    :param dtype: type to fill array with
    :return: new diagonal array
    """
    result = np.zeros((n,n), dtype=dtype)
    for i in range(n):
        result[i][i] = one

    return result


def dense_adjacency_matrix(n, dtype=np.uint32):
    """
    Return the adjacency matrix of a dense graph on n vertices

    Example:

        >>> dense_adjacency_matrix(3)
        np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]], dtype=np.uint32)

    :param n: Vertex set size
    :param dtype: Type that array should hold
    :return:
    """
    return ones((n,n), dtype) - diagonal(n, dtype=np.uint32)


def generate_lower_triangle(n, diagonal=False):
    """
    Generates the index-tuples of the lower triangle of a matrix of dimensions
    nxn. Returns a generator.

    Example:
        >>> generate_lower_triangle(3)
        <generator object generate_lower_triangle at 0x1234567890abcdef>
        >>> g = _
        >>> for x in g:
        ...    print(x)
        ...
        (1, 0)
        (2, 0)
        (2, 1)

    Note that the diagonal was not printed here. This can be included as follows

        >>> for x in generate_lower_triangle(3, diagonal=True):
        ...     print(x)
        ...
        (0, 0)
        (1, 0)
        (1, 1)
        (2, 0)
        (2, 1)
        (2, 2)

    :param n:
    :param diagonal:
    :return: A generator that generates the indices of the bottom triangle
    """
    for a in range(n):
        for b in range(a):
            yield (a, b)
        if diagonal:
            yield (a, a)

