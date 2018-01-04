"""
This module stores various mathy operations to be used throughout this package.
"""
from math import factorial


def choose(n, k):
    return factorial(n) / (factorial(k)*factorial(n - k))
