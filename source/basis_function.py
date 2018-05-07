import scipy.integrate as integrate


class MultiFunc(object):
    """Class to describe multiply two functions as a function

        Attributes:
            _func0 := first function in product
            _func1 := second function in product

        Methods:
            __init__ := constructor
            __mul__  := multiply the function by another one
            __call__ := return the result of the multiply function
            integral := return the integral of the function
    """

    def __init__(self, func0, func1):
        self._func0 = func0
        self._func1 = func1

    def __mul__(self, func):
        """Define multiplication by a function

            Args:
                func := the function to multiply by

            Returns:
                MultiFunc of self and func
        """

        return MultiFunc(self, func)

    def __call__(self, x):
        """Call the function

            Args:
                x := the input variable

            Returns:
                product of results from both functions involved
        """

        value0 = self._func0(x)
        value1 = self._func1(x)

        return value0 * value1

    def integral(self, lower, upper):
        """Integrate function from lower to upper using quadrature

            int_lower^upper self(x) dx

            Args:
                lower := lower bound for integration
                upper := upper bound for integration

            Returns:
                integral of self from lower to upper
        """

        return integrate.quad(self, lower, upper)[0]


class BasisFunction(object):
    """Class to contain a single basis function

        Attributes:
            _index      := index of the basis function
            _derivative := derivative of the basis function (0, 1, 2)
            _num_points := number of points
            _delta_x    := distance between the points

        Properties:
            delta_x_2 := _delta_x**2
            delta_x_3 := _delta_x**3
            center    := find the center of the function

        Method:
            __mul__  := multiply this by another function
            __call__ := call the function for the result
            new_x    := move input to reference interval
            in_lower := determine if in lower half of interval
            in_upper := determine if in upper half of interval

            _basis    := combine the upper and lower functions
            _lower_0  := no derivatives lower part
            _upper_0  := no derivatives upper part
            _basis_0  := no derivatives basis function
            _lower_1  := one derivatives lower part
            _upper_1  := one derivatives upper part
            _basis_1  := one derivatives basis function
            _lower_2  := two derivatives lower part
            _upper_2  := two derivatives upper part
            _basis_2  := two derivatives basis function
            _evaluate := select the basis of correct derivative
    """

    def __init__(self, index, derivative, num_points, delta_x):
        self._index = index
        self._derivative = derivative
        self._num_points = num_points
        self._delta_x = delta_x

    def __mul__(self, func):
        """Define multiplication by a function

            Args:
                func := the function to multiply by

            Returns:
                MultiFunc of self and func
        """

        return MultiFunc(self, func)

    @property
    def delta_x_2(self):
        """Find the square of delta_x

            Returns:
                delta_x squared
        """

        return self._delta_x**2

    @property
    def delta_x_3(self):
        """Find the cube of delta_x

            Returns:
                delta_x cubed
        """

        return self._delta_x**3

    @property
    def center(self):
        """Get the center of the function

            Returns:
                the center point for the function
        """

        return self._index * self._delta_x

    def new_x(self, x):
        """Map an x to the reference interval

            Returns:
                move x onto the reference interval
        """

        return x - self.center

    def in_lower(self, x):
        """Determine if x is in lower interval

            Args:
                x := the shifted input variable

            Returns:
                True:  if x is between -delta_x and 0.0
                False: otherwise
        """

        return (-self._delta_x <= x) and (x <= 0.0)

    def in_upper(self, x):
        """Determine if x is in upper interval


            Args:
                x := the shifted input variable

            Returns:
                True:  if x is between 0.0 and delta_x
                False: otherwise
        """

        return (self._delta_x >= x) and (x >= 0.0)

    def _basis(self, x, lower, upper):
        """Put basis function together

            Args:
                x     := the shifted input variable
                lower := lower function
                upper := upper function

            Returns:
                lower(x) if in lower part of interval
                upper(x) if in upper part of interval
                0.0 if not in interval
        """

        if self.in_lower(x):
            return lower(x)
        elif self.in_upper(x):
            return upper(x)
        else:
            return 0.0

    def _lower_0(self, x):
        """The lower part with derivative 0:

            Args:
                x := the shifted input variable

            Returns:
                pass
        """
        pass

    def _upper_0(self, x):
        """The upper part with derivative 0:

            Args:
                x := the shifted input variable

            Returns:
                pass
        """
        pass

    def _basis_0(self, x):
        """The basis function for derivative 0

            Args:
                x := the shifted input variable

            Returns:
                the result of the basis function with 0 derivative
        """

        return self._basis(x, self._lower_0, self._upper_0)

    def _lower_1(self, x):
        """The lower part with derivative 1:

            Args:
                x := the shifted input variable

            Returns:
                pass
        """
        pass

    def _upper_1(self, x):
        """The upper part with derivative 1:

            Args:
                x := the shifted input variable

            Returns:
                pass
        """
        pass

    def _basis_1(self, x):
        """The basis function for derivative 1

            Args:
                x := the shifted input variable

            Returns:
                the result of the basis function with 1 derivative
        """

        return self._basis(x, self._lower_1, self._upper_1)

    def _lower_2(self, x):
        """The lower part with derivative 2:

            Args:
                x := the shifted input variable

            Returns:
                pass
        """
        pass

    def _upper_2(self, x):
        """The upper part with derivative 2:

            Args:
                x := the shifted input variable

            Returns:
                pass
        """
        pass

    def _basis_2(self, x):
        """The basis function for derivative 2

            Args:
                x := the shifted input variable

            Returns:
                the result of the basis function with 2 derivative
        """

        return self._basis(x, self._lower_2, self._upper_2)

    def _evaluate(self, x):
        """Select the correct basis function via the internal derivative value

            Args:
                x := the shifted input variable

            Returns:
                result of _basis_0 if _derivative == 0
                result of _basis_1 if _derivative == 1
                result of _basis_2 if _derivative == 2
                error otherwise
        """

        if self._derivative == 0:
            return self._basis_0(x)
        elif self._derivative == 1:
            return self._basis_1(x)
        elif self._derivative == 2:
            return self._basis_2(x)
        else:
            raise AttributeError('Invalid derivative')

    def __call__(self, x):
        """Evaluate the basis function at x

            Args:
                x := input variable

            Returns:
                value of the function at x
        """

        new_x = self.new_x(x)

        return self._evaluate(new_x)


class Lagrange(BasisFunction):
    """Class to contain a single Lagrange basis function

        Inherits from Basis function
    """

    def _part0_0(self, x):
        """Evaluate first part of a Lagrange basis function with 0 derivative

            f00(x) = 2/delta_x^3 * x^3

            Args:
                x := the shifted input variable

            Returns:
                f00(x)
        """

        return (x**3 * 2.0) / self.delta_x_3

    def _part1_0(self, x):
        """Evaluate second part of a Lagrange basis function with 0 derivative

            f10(x) = 3/delta_x^2 * x^2

            Args:
                x := the shifted input variable

            Returns:
                f10(x)
        """

        return (x**2 * 3.0) / self.delta_x_2

    def _lower_0(self, x):
        """Evaluate lower part of a Lagrange basis function with 0 derivative

            fl0(x) = -f00(x) - f10(x) + 1

            Args:
                x := the shifted input variable

            Returns:
                fl0(x)
        """

        return -self._part0_0(x) - self._part1_0(x) + 1.0

    def _upper_0(self, x):
        """Evaluate upper part of a Lagrange basis function with 0 derivative

            fu0(x) = f00(x) - f10(x) + 1

            Args:
                x := the shifted input variable

            Returns:
                fu0(x)
        """

        return self._part0_0(x) - self._part1_0(x) + 1.0

    def _part0_1(self, x):
        """Evaluate first part of a Lagrange basis function with 1 derivative

            f01(x) = 6/delta_x^3 * x^2

            Args:
                x := the shifted input variable

            Returns:
                f01(x)
        """

        return (x**2 * 6.0) / self.delta_x_3

    def _part1_1(self, x):
        """Evaluate second part of a Lagrange basis function with 1 derivative

            f11(x) = 6/delta_x^2 * x

            Args:
                x := the shifted input variable

            Returns:
                f11(x)
        """

        return (x * 6.0) / self.delta_x_2

    def _lower_1(self, x):
        """Evaluate lower part of a Lagrange basis function with 1 derivative

            fl1(x) = -f01(x) - f11(x)

            Args:
                x := the shifted input variable

            Returns:
                fl1(x)
        """

        return -self._part0_1(x) - self._part1_1(x)

    def _upper_1(self, x):
        """Evaluate upper part of a Lagrange basis function with 1 derivative

            fu1(x) = f01(x) - f11(x)

            Args:
                x := the shifted input variable

            Returns:
                fu1(x)
        """

        return self._part0_1(x) - self._part1_1(x)

    def _part0_2(self, x):
        """Evaluate first part of a Lagrange basis function with 2 derivative

            f02(x) = 12/delta_x^3 * x

            Args:
                x := the shifted input variable

            Returns:
                f02(x)
        """

        return (x * 12.0) / self.delta_x_3

    def _part1_2(self, x):
        """Evaluate second part of a Lagrange basis function with 2 derivative

            f12(x) = 6/delta_x^2

            Args:
                x := the shifted input variable

            Returns:
                f12(x)
        """

        return 6.0 / self.delta_x_2

    def _lower_2(self, x):
        """Evaluate lower part of a Lagrange basis function with 2 derivative

            fl2(x) = -f02(x) - f12(x)

            Args:
                x := the shifted input variable

            Returns:
                fl2(x)
        """

        return -self._part0_2(x) - self._part1_2(x)

    def _upper_2(self, x):
        """Evaluate upper part of a Lagrange basis function with 2 derivative

            fu2(x) = f02(x) - f12(x)

            Args:
                x := the shifted input variable

            Returns:
                fu2(x)
        """

        return self._part0_2(x) - self._part1_2(x)


class Hermite(BasisFunction):
    """Class to contain a single Hermite basis function

        Inherits from Basis function
    """

    def _part0_0(self, x):
        """Evaluate first part of a Hermite basis function with 0 derivative

            f00(x) = 1/delta_x^2 * x^3

            Args:
                x := the shifted input variable

            Returns:
                f00(x)
        """

        return (x**3) / self.delta_x_2

    def _part1_0(self, x):
        """Evaluate second part of a Hermite basis function with 0 derivative

            f10(x) = 2/delta_x * x^2

            Args:
                x := the shifted input variable

            Returns:
                f10(x)
        """

        return (x**2 * 2.0) / self._delta_x

    def _lower_0(self, x):
        """Evaluate lower part of a Lagrange basis function with 0 derivative

            fl0(x) = f00(x) + f10(x) + x

            Args:
                x := the shifted input variable

            Returns:
                fl0(x)
        """

        return self._part0_0(x) + self._part1_0(x) + x

    def _upper_0(self, x):
        """Evaluate lower part of a Lagrange basis function with 0 derivative

            fu0(x) = f00(x) - f10(x) + x

            Args:
                x := the shifted input variable

            Returns:
                fu0(x)
        """

        return self._part0_0(x) - self._part1_0(x) + x

    def _part0_1(self, x):
        """Evaluate first part of a Hermite basis function with 1 derivative

            f10(x) = 3/delta_x^2 * x^2

            Args:
                x := the shifted input variable

            Returns:
                f10(x)
        """

        return (x**2 * 3.0) / self.delta_x_2

    def _part1_1(self, x):
        """Evaluate second part of a Hermite basis function with 1 derivative

            f11(x) = 4/delta_x * x

            Args:
                x := the shifted input variable

            Returns:
                f11(x)
        """

        return (x * 4.0) / self._delta_x

    def _lower_1(self, x):
        """Evaluate lower part of a Lagrange basis function with 1 derivative

            fl1(x) = f10(x) + f11(x) + 1

            Args:
                x := the shifted input variable

            Returns:
                fl1(x)
        """

        return self._part0_1(x) + self._part1_1(x) + 1.0

    def _upper_1(self, x):
        """Evaluate upper part of a Lagrange basis function with 1 derivative

            fu1(x) = f10(x) - f11(x) + 1

            Args:
                x := the shifted input variable

            Returns:
                fu1(x)
        """

        return self._part0_1(x) - self._part1_1(x) + 1.0

    def _part0_2(self, x):
        """Evaluate first part of a Hermite basis function with 2 derivative

            f20(x) = 6/delta_x^2 * x

            Args:
                x := the shifted input variable

            Returns:
                f20(x)
        """

        return (x * 6.0) / self.delta_x_2

    def _part1_2(self, x):
        """Evaluate second part of a Hermite basis function with 2 derivative

            f21(x) = 4/delta_x

            Args:
                x := the shifted input variable

            Returns:
                f21(x)
        """

        return 4.0 / self._delta_x

    def _lower_2(self, x):
        """Evaluate lower part of a Lagrange basis function with 2 derivative

            fl2(x) = f20(x) + f21(x)

            Args:
                x := the shifted input variable

            Returns:
                fl1(x)
        """

        return self._part0_2(x) + self._part1_2(x)

    def _upper_2(self, x):
        """Evaluate upper part of a Lagrange basis function with 2 derivative

            fu2(x) = f20(x) - f21(x)

            Args:
                x := the shifted input variable

            Returns:
                fu2(x)
        """

        return self._part0_2(x) - self._part1_2(x)
