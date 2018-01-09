"""
    act.actions
    ~~~~~~~~~~~

    This defines group actions on sets/graphs. This currently uses SymPy
    `Permutation`s and `PermutationGroup`.
"""


from sympy.combinatorics import Permutation, PermutationGroup

from agt.graph import MatrixGraph


class GroupAction:
    def __init__(self, group, space=None):
        """
        Create a new GroupAction
        :param group: `group` is either a SymPy Permutation, a list of SymPy
            Permutations, or a SymPy PermutationGroup.
        :param space: `space` is a set {0,1,2,...,n-1} that specifies the
            set to be acted upon.
        """

        self.group = None
        if isinstance(group, PermutationGroup):
            self.group = group
        elif isinstance(group, Permutation):
            self.group = PermutationGroup([group])
        elif isinstance(group, list):
            if len(group) > 0:
                self.group = PermutationGroup(group)
            else:
                self.group = PermutationGroup()

    def __call__(self, item):
        if isinstance(item, MatrixGraph):
            pass
