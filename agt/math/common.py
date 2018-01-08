"""
This module stores various mathy operations to be used throughout this package.
"""
from math import factorial


def choose(n, k):
    """
    Compute nCk
    :param n:
    :param k:
    :return:
    """
    return factorial(n) // (factorial(k)*factorial(n - k))


def gcd(a, b):
    """
    Compute the greatest common divisor of a and b
    :param a:
    :param b:
    :return:
    """
    if a == 0:
        return b
    return gcd(b % a, a)


def lcm(a, b):
    """
    Compute the least common multiple of a and b
    :param a:
    :param b:
    :return:
    """

    if a == b == 0:
        return 0

    return (a * b) // gcd(a, b)

