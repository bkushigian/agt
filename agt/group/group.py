class Group:
    def order(self):
        """Return order of group: n if finite, math.inf otherwise"""
        raise NotImplementedError()

    def identity(self):
        """Return the identity element of G"""
        raise NotImplementedError()

    def inverse(self, x):
        """Return x' such that x'x = xx' = e"""
        raise NotImplementedError()

    def abelian(self):
        """Return True if this is Abelian, False otherwise"""
        raise NotImplementedError()

    def center(self):
        """Return the center of G"""
        raise NotImplementedError()

    def is_subgroup(self, H):
        """Return True if self is a subgroup of H, False otherwise"""
        raise NotImplementedError()
