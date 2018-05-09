class InitV(object):
    """Class to contain the initial function for v"""

    def __call__(self, x):
        """Run the function"""

        # return 5.0 * x**2
        return 0.01 * x**2


class InitW(object):
    """Class to contain the initial function for w"""

    def __call__(self, x):
        """Run the function"""

        # return 5.0 * x**3
        return 0.01 * x**3


class InitZ0(object):
    """Class to contain the initial function for z0"""

    def __call__(self, x):
        """Run the function"""

        # return x**3
        return 0.01 * x**3


class InitZ1(object):
    """Class to contain the initial function for z1"""

    def __call__(self, x):
        """Run the function"""

        # return x
        return 0.01 * x


class G1(object):
    """Class to contain the g1 control function"""

    def __call__(self, x):
        """Run the function"""

        return x


class G2(object):
    """Class to contain the g2 control function"""

    def __call__(self, x):
        """Run the function"""

        return x


class M(object):
    """Class to contain the m control function"""

    def __call__(self, x):
        """Run the function"""

        return x
