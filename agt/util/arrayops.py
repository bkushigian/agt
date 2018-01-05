"""
arrayops stores various array operations that will come in handy
"""

import bisect
import numpy as np

zero = np.uint32(0)
one = np.uint32(1)


def index(a, x):
    """Locate the leftmost value exactly equal to x"""
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise None


def ones(shape, dtype=np.uint32):
    """Create a numpy array of ones (like np.zeros:)"""
    return np.full(shape, one, dtype=dtype)


def diagonal(shape, dtype=np.uint32):
    """Create a diagonal matrix"""
    assert shape[0] == shape[1]
    result = np.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        result[i][i] = one

    return result


def dense_adjacency_matrix(shape, dtype=np.uint32):
    return ones(shape, dtype) - diagonal(shape, dtype=np.uint32)


def generate_lower_triangle(size, diagonal=False):
    """Generate tuples (a,b) such that 0 <= a < size and b < a"""
    for a in range(size):
        for b in range(a):
            yield (a,b)
        if diagonal:
            yield (a,a)

