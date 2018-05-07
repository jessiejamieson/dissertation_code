import numpy as np

from source import basis_function as basis


class Element(object):
    """Class to contain a single element and its related matrices

        Attributes:
            _index      := index
            _num_points := num_points
            _delta_x    := delta_x

            LL_00 := Lagrange_0_diff-Lagrange_0_diff
            LL_01 := Lagrange_0_diff-Lagrange_1_diff
            LL_02 := Lagrange_0_diff-Lagrange_2_diff
            LL_10 := Lagrange_1_diff-Lagrange_0_diff
            LL_11 := Lagrange_1_diff-Lagrange_1_diff
            LL_12 := Lagrange_1_diff-Lagrange_2_diff
            LL_20 := Lagrange_2_diff-Lagrange_0_diff
            LL_21 := Lagrange_2_diff-Lagrange_1_diff
            LL_22 := Lagrange_2_diff-Lagrange_2_diff

            LH_00 := Lagrange_0_diff-Hermite_0_diff
            LH_01 := Lagrange_0_diff-Hermite_1_diff
            LH_02 := Lagrange_0_diff-Hermite_2_diff
            LH_10 := Lagrange_1_diff-Hermite_0_diff
            LH_11 := Lagrange_1_diff-Hermite_1_diff
            LH_12 := Lagrange_1_diff-Hermite_2_diff
            LH_20 := Lagrange_2_diff-Hermite_0_diff
            LH_21 := Lagrange_2_diff-Hermite_1_diff
            LH_22 := Lagrange_2_diff-Hermite_2_diff

            HL_00 := Hermite_0_diff-Lagrange_0_diff
            HL_01 := Hermite_0_diff-Lagrange_1_diff
            HL_02 := Hermite_0_diff-Lagrange_2_diff
            HL_10 := Hermite_1_diff-Lagrange_0_diff
            HL_11 := Hermite_1_diff-Lagrange_1_diff
            HL_12 := Hermite_1_diff-Lagrange_2_diff
            HL_20 := Hermite_2_diff-Lagrange_0_diff
            HL_21 := Hermite_2_diff-Lagrange_1_diff
            HL_22 := Hermite_2_diff-Lagrange_2_diff

            HH_00 := Hermite_0_diff-Hermite_0_diff
            HH_01 := Hermite_0_diff-Hermite_1_diff
            HH_02 := Hermite_0_diff-Hermite_2_diff
            HH_10 := Hermite_1_diff-Hermite_0_diff
            HH_11 := Hermite_1_diff-Hermite_1_diff
            HH_12 := Hermite_1_diff-Hermite_2_diff
            HH_20 := Hermite_2_diff-Hermite_0_diff
            HH_21 := Hermite_2_diff-Hermite_1_diff
            HH_22 := Hermite_2_diff-Hermite_2_diff

            LLL_110 := Lagrange_1-Lagrange_1-Lagrange_0
            LLL_200 := Lagrange_1-Lagrange_0-Lagrange_1

            LLH_110 := Lagrange_1-Lagrange_1-Hermite_0
            LLH_200 := Lagrange_1-Lagrange_0-Hermite_1

            LHL_110 := Lagrange_1-Hermite_1-Lagrange_0
            LHL_200 := Lagrange_1-Hermite_0-Lagrange_1

            LHH_110 := Lagrange_1-Hermite_1-Hermite_0
            LHH_200 := Lagrange_1-Hermite_0-Hermite_1

            HLL_110 := Hermite_1-Lagrange_1-Lagrange_0
            HLL_200 := Hermite_1-Lagrange_0-Lagrange_1

            HLH_110 := Hermite_1-Lagrange_1-Hermite_0
            HLH_200 := Hermite_1-Lagrange_0-Hermite_1

            HHL_110 := Hermite_1-Hermite_1-Lagrange_0
            HHL_200 := Hermite_1-Hermite_0-Lagrange_1

            HHH_110 := Hermite_1-Hermite_1-Hermite_0
            HHH_200 := Hermite_1-Hermite_0-Hermite_1
    """

    def __init__(self, index, num_points, delta_x):
        self._index = index
        self._num_points = num_points
        self._delta_x = delta_x

    @classmethod
    def setup(cls, index, num_points, delta_x):
        """Setup the element

            Args:
                index      := index
                num_points := num_points
                delta_x    := delta_x

            Returns:
                fully built element
        """

        new_element = cls(index, num_points, delta_x)
        new_element._build()

        return new_element

    @property
    def lower_bound(self):
        """Get the lower bound of interval

            Returns:
                lower bound of interval
        """

        return self._index * self._delta_x

    @property
    def upper_bound(self):
        """Get the upper bound of interval

            Returns:
                upper bound of interval
        """

        return (self._index + 1) * self._delta_x

    @property
    def basis_index(self):
        """Get the indices for nonzero basis functions on element

            Returns:
                list if indices for non-zero functions
        """

        return [self._index, (self._index + 1)]

    def build_lagrange(self, index, derivative):
        """Build a Lagrange basis function

            Args:
                index      := index
                derivative := derivative

            Returns:
                the Lagrange basis function with index and derivative
        """

        return basis.Lagrange(index, derivative,
                              self._num_points, self._delta_x)

    def build_hermite(self, index, derivative):
        """Build a Hermite basis function

            Args:
                index      := index
                derivative := derivative

            Returns:
                the Hermite Lagrange basis function with index and derivative
        """

        return basis.Hermite(index, derivative,
                             self._num_points, self._delta_x)

    def func_pl(self, p, index0, diff0):
        """Find the P-lagrange product

            f = L(index0, diff0) * p

            Args:
                p      := the function for product
                index0 := index of basis function
                diff0  := derivative of basis function

            Returns:
                product of Lagrange basis with p i.e. f
        """

        function0 = self.build_lagrange(index0, diff0)
        function1 = p

        return function0 * function1

    def func_ph(self, p, index0, diff0):
        """Find the P-hermite product

            g = H(index0, diff0) * p

            Args:
                p      := the function for product
                index0 := index of basis function
                diff0  := derivative of basis function

            Returns:
                product of Hermite basis with p i.e. g
        """

        function0 = self.build_hermite(index0, diff0)
        function1 = p

        return function0 * function1

    def func_ll(self, index0, index1, diff0, diff1):
        """Find the Lagrange-Lagrange product

            fll = L(index0, diff0) * L(index1, diff1)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function

            Returns:
                product of the basis functions i.e. fll
        """

        function0 = self.build_lagrange(index0, diff0)
        function1 = self.build_lagrange(index1, diff1)

        return function0 * function1

    def func_lh(self, index0, index1, diff0, diff1):
        """Find the Lagrange-Hermite product

            flh = L(index0, diff0) * H(index1, diff1)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function

            Returns:
                product of the basis functions i.e. flh
        """

        function0 = self.build_lagrange(index0, diff0)
        function1 = self.build_hermite(index1, diff1)

        return function0 * function1

    def func_hl(self, index0, index1, diff0, diff1):
        """Find the Hermite-Lagrange product

            fhl = H(index0, diff0) * L(index1, diff1)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function

            Returns:
                product of the basis functions i.e. fhl
        """

        function0 = self.build_hermite(index0, diff0)
        function1 = self.build_lagrange(index1, diff1)

        return function0 * function1

    def func_hh(self, index0, index1, diff0, diff1):
        """Find the Hermite-Hermite product

            fhh = H(index0, diff0) * H(index1, diff1)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function

            Returns:
                product of the basis functions i.e. fhh
        """

        function0 = self.build_hermite(index0, diff0)
        function1 = self.build_hermite(index1, diff1)

        return function0 * function1

    def func_lll(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Lagrange-Lagrange-Lagrange product

            f_lll = L(index0, diff0) * L(index1, diff1) * L(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_lll
        """

        function0 = self.func_ll(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_llh(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Lagrange-Lagrange-Hermite product

            f_llh = L(index0, diff0) * L(index1, diff1) * H(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_llh
        """

        function0 = self.func_ll(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def func_lhl(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Lagrange-Hermite-Lagrange product

            f_lhl = L(index0, diff0) * H(index1, diff1) * L(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_lhl
        """

        function0 = self.func_lh(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_lhh(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Lagrange-Hermite-Hermite product

            f_lhh = L(index0, diff0) * H(index1, diff1) * H(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_lhh
        """

        function0 = self.func_lh(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def func_hll(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Hermite-Lagrange-Lagrange product

            f_hll = H(index0, diff0) * L(index1, diff1) * L(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_hll
        """

        function0 = self.func_hl(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_hlh(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Hermite-Lagrange-Hermite product

            f_hlh = H(index0, diff0) * L(index1, diff1) * H(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_hlh
        """

        function0 = self.func_hl(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def func_hhl(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Hermite-Hermite-Lagrange product

            f_hhl = H(index0, diff0) * H(index1, diff1) * L(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_hhl
        """

        function0 = self.func_hh(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_hhh(self, index0, index1, index2, diff0, diff1, diff2):
        """Find the Hermite-Hermite-Hermite product

            f_hhh = H(index0, diff0) * H(index1, diff1) * H(index2, diff2)

            Args:
                index0 := index of first basis function
                index1 := index of second basis function
                index2 := index of third basis function
                diff0  := derivative of first basis function
                diff1  := derivative of second basis function
                diff2  := derivative of third basis function

            Returns:
                product of the basis functions i.e. f_hhh
        """

        function0 = self.func_hh(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def pl_2(self, p):
        """Build the PL_2 array

            PL_2 = [L(0, 2) * p, L(1, 2)]

            Args:
                p := the function to build array

            Returns:
                array of products with Lagrange, i.e. PL_2
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func0 = self.func_pl(p, basis_index0, 2)
        func1 = self.func_pl(p, basis_index1, 2)

        point_0 = func0.integral(self.lower_bound, self.upper_bound)
        point_1 = func1.integral(self.lower_bound, self.upper_bound)

        return np.array([point_0, point_1])

    def ph_2(self, p):
        """Build the PH_2 array

            PH_2 = [H(0, 2) * p, H(1, 2)]

            Args:
                p := the function to build array

            Returns:
                array of products with Hermite, i.e. PH_2
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func0 = self.func_ph(p, basis_index0, 2)
        func1 = self.func_ph(p, basis_index1, 2)

        point_0 = func0.integral(self.lower_bound, self.upper_bound)
        point_1 = func1.integral(self.lower_bound, self.upper_bound)

        return np.array([point_0, point_1])

    def _build_ll_00(self):
        """Build the LL_00 matrix

            LL_00 = [[L(0, 0) * L(0, 0), L(0, 0) * L(1, 0)],
                     [L(1, 0) * L(0, 0), L(1, 0) * L(1, 0)]]

            Sets:
                LL_00

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 0, 0)
        func_01 = self.func_ll(basis_index0, basis_index1, 0, 0)
        func_10 = self.func_ll(basis_index1, basis_index0, 0, 0)
        func_11 = self.func_ll(basis_index1, basis_index1, 0, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_00 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_01(self):
        """Build the LL_01 matrix

            LL_01 = [[L(0, 0) * L(0, 1), L(0, 0) * L(1, 1)],
                     [L(1, 0) * L(0, 1), L(1, 0) * L(1, 1)]]

            Sets:
                LL_01

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 0, 1)
        func_01 = self.func_ll(basis_index0, basis_index1, 0, 1)
        func_10 = self.func_ll(basis_index1, basis_index0, 0, 1)
        func_11 = self.func_ll(basis_index1, basis_index1, 0, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_01 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_02(self):
        """Build the LL_02 matrix

            LL_02 = [[L(0, 0) * L(0, 2), L(0, 0) * L(1, 2)],
                     [L(1, 0) * L(0, 2), L(1, 0) * L(1, 2)]]

            Sets:
                LL_02

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 0, 2)
        func_01 = self.func_ll(basis_index0, basis_index1, 0, 2)
        func_10 = self.func_ll(basis_index1, basis_index0, 0, 2)
        func_11 = self.func_ll(basis_index1, basis_index1, 0, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_02 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_10(self):
        """Build the LL_10 matrix

            LL_10 = [[L(0, 1) * L(0, 0), L(0, 1) * L(1, 0)],
                     [L(1, 1) * L(0, 0), L(1, 1) * L(1, 0)]]

            Sets:
                LL_10

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 1, 0)
        func_01 = self.func_ll(basis_index0, basis_index1, 1, 0)
        func_10 = self.func_ll(basis_index1, basis_index0, 1, 0)
        func_11 = self.func_ll(basis_index1, basis_index1, 1, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_10 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_11(self):
        """Build the LL_11 matrix

            LL_11 = [[L(0, 1) * L(0, 1), L(1, 1) * L(1, 1)],
                     [L(1, 1) * L(0, 1), L(1, 1) * L(1, 1)]]

            Sets:
                LL_11

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 1, 1)
        func_01 = self.func_ll(basis_index0, basis_index1, 1, 1)
        func_10 = self.func_ll(basis_index1, basis_index0, 1, 1)
        func_11 = self.func_ll(basis_index1, basis_index1, 1, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_11 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_12(self):
        """Build the LL_12 matrix

            LL_12 = [[L(0, 1) * L(0, 2), L(0, 1) * L(1, 2)],
                     [L(1, 1) * L(0, 2), L(1, 1) * L(1, 2)]]

            Sets:
                LL_12

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 1, 2)
        func_01 = self.func_ll(basis_index0, basis_index1, 1, 2)
        func_10 = self.func_ll(basis_index1, basis_index0, 1, 2)
        func_11 = self.func_ll(basis_index1, basis_index1, 1, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_12 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_20(self):
        """Build the LL_20 matrix

            LL_20 = [[L(0, 2) * L(0, 0), L(0, 2) * L(1, 0)],
                     [L(1, 2) * L(0, 0), L(1, 2) * L(1, 0)]]

            Sets:
                LL_20

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 2, 0)
        func_01 = self.func_ll(basis_index0, basis_index1, 2, 0)
        func_10 = self.func_ll(basis_index1, basis_index0, 2, 0)
        func_11 = self.func_ll(basis_index1, basis_index1, 2, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_20 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_21(self):
        """Build the LL_21 matrix

            LL_21 = [[L(0, 2) * L(0, 1), L(0, 2) * L(1, 1)],
                     [L(1, 2) * L(0, 1), L(1, 2) * L(1, 1)]]

            Sets:
                LL_21

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 2, 1)
        func_01 = self.func_ll(basis_index0, basis_index1, 2, 1)
        func_10 = self.func_ll(basis_index1, basis_index0, 2, 1)
        func_11 = self.func_ll(basis_index1, basis_index1, 2, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_21 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_ll_22(self):
        """Build the LL_22 matrix

            LL_22 = [[L(0, 2) * L(0, 2), L(0, 2) * L(1, 2)],
                     [L(1, 2) * L(0, 2), L(1, 2) * L(1, 2)]]

            Sets:
                LL_22

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_ll(basis_index0, basis_index0, 2, 2)
        func_01 = self.func_ll(basis_index0, basis_index1, 2, 2)
        func_10 = self.func_ll(basis_index1, basis_index0, 2, 2)
        func_11 = self.func_ll(basis_index1, basis_index1, 2, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LL_22 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_00(self):
        """Build the LH_00 matrix

            LH_00 = [[L(0, 0) * H(0, 0), L(0, 0) * H(1, 0)],
                     [L(1, 0) * H(0, 0), L(1, 0) * H(1, 0)]]

            Sets:
                LH_00

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 0, 0)
        func_01 = self.func_lh(basis_index0, basis_index1, 0, 0)
        func_10 = self.func_lh(basis_index1, basis_index0, 0, 0)
        func_11 = self.func_lh(basis_index1, basis_index1, 0, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_00 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_01(self):
        """Build the LH_01 matrix

            LH_01 = [[L(0, 0) * H(0, 1), L(0, 0) * H(1, 1)],
                     [L(1, 0) * H(0, 1), L(1, 0) * H(1, 1)]]

            Sets:
                LH_01

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 0, 1)
        func_01 = self.func_lh(basis_index0, basis_index1, 0, 1)
        func_10 = self.func_lh(basis_index1, basis_index0, 0, 1)
        func_11 = self.func_lh(basis_index1, basis_index1, 0, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_01 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_02(self):
        """Build the LH_02 matrix

            LH_02 = [[L(0, 0) * H(0, 2), L(0, 0) * H(1, 2)],
                     [L(1, 0) * H(0, 2), L(1, 0) * H(1, 2)]]

            Sets:
                LH_02

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 0, 2)
        func_01 = self.func_lh(basis_index0, basis_index1, 0, 2)
        func_10 = self.func_lh(basis_index1, basis_index0, 0, 2)
        func_11 = self.func_lh(basis_index1, basis_index1, 0, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_02 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_10(self):
        """Build the LH_10 matrix

            LH_10 = [[L(0, 1) * H(0, 0), L(0, 1) * H(1, 0)],
                     [L(1, 1) * H(0, 0), L(1, 1) * H(1, 0)]]

            Sets:
                LH_10

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 1, 0)
        func_01 = self.func_lh(basis_index0, basis_index1, 1, 0)
        func_10 = self.func_lh(basis_index1, basis_index0, 1, 0)
        func_11 = self.func_lh(basis_index1, basis_index1, 1, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_10 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_11(self):
        """Build the LH_11 matrix

            LH_11 = [[L(0, 1) * H(0, 1), L(0, 1) * H(1, 1)],
                     [L(1, 1) * H(0, 1), L(1, 1) * H(1, 1)]]

            Sets:
                LH_11

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 1, 1)
        func_01 = self.func_lh(basis_index0, basis_index1, 1, 1)
        func_10 = self.func_lh(basis_index1, basis_index0, 1, 1)
        func_11 = self.func_lh(basis_index1, basis_index1, 1, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_11 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_12(self):
        """Build the LH_12 matrix

            LH_12 = [[L(0, 1) * H(0, 2), L(0, 1) * H(1, 2)],
                     [L(1, 1) * H(0, 2), L(1, 1) * H(1, 2)]]

            Sets:
                LH_12

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 1, 2)
        func_01 = self.func_lh(basis_index0, basis_index1, 1, 2)
        func_10 = self.func_lh(basis_index1, basis_index0, 1, 2)
        func_11 = self.func_lh(basis_index1, basis_index1, 1, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_12 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_20(self):
        """Build the LH_20 matrix

            LH_20 = [[L(0, 2) * H(0, 0), L(0, 2) * H(1, 0)],
                     [L(1, 2) * H(0, 0), L(1, 2) * H(1, 0)]]

            Sets:
                LH_20

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 2, 0)
        func_01 = self.func_lh(basis_index0, basis_index1, 2, 0)
        func_10 = self.func_lh(basis_index1, basis_index0, 2, 0)
        func_11 = self.func_lh(basis_index1, basis_index1, 2, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_20 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_21(self):
        """Build the LH_21 matrix

            LH_21 = [[L(0, 2) * H(0, 1), L(0, 2) * H(1, 1)],
                     [L(1, 2) * H(0, 1), L(1, 2) * H(1, 1)]]

            Sets:
                LH_21

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 2, 1)
        func_01 = self.func_lh(basis_index0, basis_index1, 2, 1)
        func_10 = self.func_lh(basis_index1, basis_index0, 2, 1)
        func_11 = self.func_lh(basis_index1, basis_index1, 2, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_21 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lh_22(self):
        """Build the LH_22 matrix

            LH_22 = [[L(0, 2) * H(0, 2), L(0, 2) * H(1, 2)],
                     [L(1, 2) * H(0, 2), L(1, 2) * H(1, 2)]]

            Sets:
                LH_22

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_lh(basis_index0, basis_index0, 2, 2)
        func_01 = self.func_lh(basis_index0, basis_index1, 2, 2)
        func_10 = self.func_lh(basis_index1, basis_index0, 2, 2)
        func_11 = self.func_lh(basis_index1, basis_index1, 2, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.LH_22 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_00(self):
        """Build the HL_00 matrix

            HL_00 = [[H(0, 0) * L(0, 0), H(0, 0) * L(1, 0)],
                     [H(1, 0) * L(0, 0), H(1, 0) * L(1, 0)]]

            Sets:
                HL_00

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 0, 0)
        func_01 = self.func_hl(basis_index0, basis_index1, 0, 0)
        func_10 = self.func_hl(basis_index1, basis_index0, 0, 0)
        func_11 = self.func_hl(basis_index1, basis_index1, 0, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_00 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_01(self):
        """Build the HL_01 matrix

            HL_01 = [[H(0, 0) * L(0, 1), H(0, 0) * L(1, 1)],
                     [H(1, 0) * L(0, 1), H(1, 0) * L(1, 1)]]

            Sets:
                HL_01

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 0, 1)
        func_01 = self.func_hl(basis_index0, basis_index1, 0, 1)
        func_10 = self.func_hl(basis_index1, basis_index0, 0, 1)
        func_11 = self.func_hl(basis_index1, basis_index1, 0, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_01 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_02(self):
        """Build the HL_02 matrix

            HL_02 = [[H(0, 0) * L(0, 2), H(0, 0) * L(1, 2)],
                     [H(1, 0) * L(0, 2), H(1, 0) * L(1, 2)]]

            Sets:
                HL_02

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 0, 2)
        func_01 = self.func_hl(basis_index0, basis_index1, 0, 2)
        func_10 = self.func_hl(basis_index1, basis_index0, 0, 2)
        func_11 = self.func_hl(basis_index1, basis_index1, 0, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_02 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_10(self):
        """Build the HL_10 matrix

            HL_10 = [[H(0, 1) * L(0, 0), H(0, 1) * L(1, 0)],
                     [H(1, 1) * L(0, 0), H(1, 1) * L(1, 0)]]

            Sets:
                HL_10

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 1, 0)
        func_01 = self.func_hl(basis_index0, basis_index1, 1, 0)
        func_10 = self.func_hl(basis_index1, basis_index0, 1, 0)
        func_11 = self.func_hl(basis_index1, basis_index1, 1, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_10 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_11(self):
        """Build the HL_11 matrix

            HL_11 = [[H(0, 1) * L(0, 1), H(0, 1) * L(1, 1)],
                     [H(1, 1) * L(0, 1), H(1, 1) * L(1, 1)]]

            Sets:
                HL_11

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 1, 1)
        func_01 = self.func_hl(basis_index0, basis_index1, 1, 1)
        func_10 = self.func_hl(basis_index1, basis_index0, 1, 1)
        func_11 = self.func_hl(basis_index1, basis_index1, 1, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_11 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_12(self):
        """Build the HL_12 matrix

            HL_12 = [[H(0, 1) * L(0, 2), H(0, 1) * L(1, 2)],
                     [H(1, 1) * L(0, 2), H(1, 1) * L(1, 2)]]

            Sets:
                HL_12

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 1, 2)
        func_01 = self.func_hl(basis_index0, basis_index1, 1, 2)
        func_10 = self.func_hl(basis_index1, basis_index0, 1, 2)
        func_11 = self.func_hl(basis_index1, basis_index1, 1, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_12 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_20(self):
        """Build the HL_20 matrix

            HL_20 = [[H(0, 2) * L(0, 0), H(0, 2) * L(1, 0)],
                     [H(1, 2) * L(0, 0), H(1, 2) * L(1, 0)]]

            Sets:
                HL_20

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 2, 0)
        func_01 = self.func_hl(basis_index0, basis_index1, 2, 0)
        func_10 = self.func_hl(basis_index1, basis_index0, 2, 0)
        func_11 = self.func_hl(basis_index1, basis_index1, 2, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_20 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_21(self):
        """Build the HL_21 matrix

            HL_21 = [[H(0, 2) * L(0, 1), H(0, 2) * L(1, 1)],
                     [H(1, 2) * L(0, 1), H(1, 2) * L(1, 1)]]

            Sets:
                HL_21

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 2, 1)
        func_01 = self.func_hl(basis_index0, basis_index1, 2, 1)
        func_10 = self.func_hl(basis_index1, basis_index0, 2, 1)
        func_11 = self.func_hl(basis_index1, basis_index1, 2, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_21 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hl_22(self):
        """Build the HL_22 matrix

            HL_22 = [[H(0, 2) * L(0, 2), H(0, 2) * L(1, 2)],
                     [H(1, 2) * L(0, 2), H(1, 2) * L(1, 2)]]

            Sets:
                HL_22

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hl(basis_index0, basis_index0, 2, 2)
        func_01 = self.func_hl(basis_index0, basis_index1, 2, 2)
        func_10 = self.func_hl(basis_index1, basis_index0, 2, 2)
        func_11 = self.func_hl(basis_index1, basis_index1, 2, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HL_22 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_00(self):
        """Build the HH_00 matrix

            HH_00 = [[H(0, 0) * H(0, 0), H(0, 0) * H(1, 0)],
                     [H(1, 0) * H(0, 0), H(1, 0) * H(1, 0)]]

            Sets:
                HH_00

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 0, 0)
        func_01 = self.func_hh(basis_index0, basis_index1, 0, 0)
        func_10 = self.func_hh(basis_index1, basis_index0, 0, 0)
        func_11 = self.func_hh(basis_index1, basis_index1, 0, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_00 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_01(self):
        """Build the HH_01 matrix

            HH_01 = [[H(0, 0) * H(0, 1), H(0, 0) * H(1, 1)],
                     [H(1, 0) * H(0, 1), H(1, 0) * H(1, 1)]]

            Sets:
                HH_01

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 0, 1)
        func_01 = self.func_hh(basis_index0, basis_index1, 0, 1)
        func_10 = self.func_hh(basis_index1, basis_index0, 0, 1)
        func_11 = self.func_hh(basis_index1, basis_index1, 0, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_01 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_02(self):
        """Build the HH_02 matrix

            HH_02 = [[H(0, 0) * H(0, 2), H(0, 0) * H(1, 2)],
                     [H(1, 0) * H(0, 2), H(1, 0) * H(1, 2)]]

            Sets:
                HH_02

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 0, 2)
        func_01 = self.func_hh(basis_index0, basis_index1, 0, 2)
        func_10 = self.func_hh(basis_index1, basis_index0, 0, 2)
        func_11 = self.func_hh(basis_index1, basis_index1, 0, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_02 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_10(self):
        """Build the HH_10 matrix

            HH_10 = [[H(0, 1) * H(0, 0), H(0, 1) * H(1, 0)],
                     [H(1, 1) * H(0, 0), H(1, 1) * H(1, 0)]]

            Sets:
                HH_10

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 1, 0)
        func_01 = self.func_hh(basis_index0, basis_index1, 1, 0)
        func_10 = self.func_hh(basis_index1, basis_index0, 1, 0)
        func_11 = self.func_hh(basis_index1, basis_index1, 1, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_10 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_11(self):
        """Build the HH_11 matrix

            HH_11 = [[H(0, 1) * H(0, 1), H(0, 1) * H(1, 1)],
                     [H(1, 1) * H(0, 1), H(1, 1) * H(1, 1)]]

            Sets:
                HH_11

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 1, 1)
        func_01 = self.func_hh(basis_index0, basis_index1, 1, 1)
        func_10 = self.func_hh(basis_index1, basis_index0, 1, 1)
        func_11 = self.func_hh(basis_index1, basis_index1, 1, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_11 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_12(self):
        """Build the HH_12 matrix

            HH_12 = [[H(0, 1) * H(0, 2), H(0, 1) * H(1, 2)],
                     [H(1, 1) * H(0, 2), H(1, 1) * H(1, 2)]]

            Sets:
                HH_12

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 1, 2)
        func_01 = self.func_hh(basis_index0, basis_index1, 1, 2)
        func_10 = self.func_hh(basis_index1, basis_index0, 1, 2)
        func_11 = self.func_hh(basis_index1, basis_index1, 1, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_12 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_20(self):
        """Build the HH_20 matrix

            HH_20 = [[H(0, 2) * H(0, 0), H(0, 2) * H(1, 0)],
                     [H(1, 2) * H(0, 0), H(1, 2) * H(1, 0)]]

            Sets:
                HH_20

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 2, 0)
        func_01 = self.func_hh(basis_index0, basis_index1, 2, 0)
        func_10 = self.func_hh(basis_index1, basis_index0, 2, 0)
        func_11 = self.func_hh(basis_index1, basis_index1, 2, 0)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_20 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_21(self):
        """Build the HH_22 matrix

            HH_22 = [[H(0, 2) * H(0, 2), H(0, 2) * H(1, 2)],
                     [H(1, 2) * H(0, 2), H(1, 2) * H(1, 2)]]

            Sets:
                HH_22

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 2, 1)
        func_01 = self.func_hh(basis_index0, basis_index1, 2, 1)
        func_10 = self.func_hh(basis_index1, basis_index0, 2, 1)
        func_11 = self.func_hh(basis_index1, basis_index1, 2, 1)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_21 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_hh_22(self):
        """Build the HH_22 matrix

            HH_22 = [[H(0, 2) * H(0, 2), H(0, 2) * H(1, 2)],
                     [H(1, 2) * H(0, 2), H(1, 2) * H(1, 2)]]

            Sets:
                HH_22

            Returns:
                None
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_00 = self.func_hh(basis_index0, basis_index0, 2, 2)
        func_01 = self.func_hh(basis_index0, basis_index1, 2, 2)
        func_10 = self.func_hh(basis_index1, basis_index0, 2, 2)
        func_11 = self.func_hh(basis_index1, basis_index1, 2, 2)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        self.HH_22 = np.array([[point_00, point_01],
                               [point_10, point_11]])

    def _build_lll_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the LLL matrix for a fixed k

            LLL = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = L(i, diff_i) * L(j, diff_j) * L(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix LLL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_lll(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_lll(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_lll(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_lll(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_llh_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the LLH matrix for a fixed k

            LLH = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = L(i, diff_i) * L(j, diff_j) * H(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix LLH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_llh(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_llh(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_llh(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_llh(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_lhl_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the LHL matrix for a fixed k

            LHL = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = L(i, diff_i) * H(j, diff_j) * L(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix LHL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_lhl(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_lhl(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_lhl(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_lhl(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_lhh_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the LHH matrix for a fixed k

            LHH = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = L(i, diff_i) * H(j, diff_j) * H(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix LHH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_lhh(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_lhh(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_lhh(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_lhh(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_hll_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the HLL matrix for a fixed k

            HLL = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = H(i, diff_i) * L(j, diff_j) * L(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix HLL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_hll(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_hll(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_hll(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_hll(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_hlh_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the HLH matrix for a fixed k

            HLH = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = H(i, diff_i) * L(j, diff_j) * H(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix HLH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_hlh(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_hlh(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_hlh(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_hlh(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_hhl_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the HHL matrix for a fixed k

            HHL = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = H(i, diff_i) * H(j, diff_j) * L(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix HHL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_hhl(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_hhl(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_hhl(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_hhl(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_hhh_matrix(self, index_k, diff_i, diff_j, diff_k):
        """Builds the HHH matrix for a fixed k

            HHH = [[Lt(0, 0), Lt(0, 1)],
                   [Lt(1, 0), Lt(1, 1)]]

            Where,

            Lt(i, j) = H(i, diff_i) * H(j, diff_j) * H(index_k, diff_k)

            Args:
                index_k := index on third basis function
                diff_i  := diff on first basis
                diff_j  := diff on second basis
                diff_k  := diff on third basis

            Returns:
                matrix HHH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        func_00 = self.func_hhh(index0, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_01 = self.func_hhh(index0, index1, index_k,
                                diff_i, diff_j, diff_k)
        func_10 = self.func_hhh(index1, index0, index_k,
                                diff_i, diff_j, diff_k)
        func_11 = self.func_hhh(index1, index1, index_k,
                                diff_i, diff_j, diff_k)

        point_00 = func_00.integral(self.lower_bound, self.upper_bound)
        point_01 = func_01.integral(self.lower_bound, self.upper_bound)
        point_10 = func_10.integral(self.lower_bound, self.upper_bound)
        point_11 = func_11.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_00, point_01],
                         [point_10, point_11]])

    def _build_lll(self, diff_i, diff_j, diff_k):
        """Builds a LLL tensor

            T_LLL = [LLL(0), LLL(1)]

            Where,

            LLL(i) = LLL(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_LLL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_lll_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_lll_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_llh(self, diff_i, diff_j, diff_k):
        """Builds a LLH tensor

            T_LLH = [LLH(0), LLH(1)]

            Where,

            LLH(i) = LLH(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_LLH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_llh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_llh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_lhl(self, diff_i, diff_j, diff_k):
        """Builds a LHL tensor

            T_LHL = [LHL(0), LHL(1)]

            Where,

            LHL(i) = LHL(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_LHL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_lhl_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_lhl_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_lhh(self, diff_i, diff_j, diff_k):
        """Builds a LHH tensor

            T_LHH = [LHH(0), LHH(1)]

            Where,

            LHH(i) = LHH(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_LHH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_lhh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_lhh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hll(self, diff_i, diff_j, diff_k):
        """Builds a HLL tensor

            T_HLL = [HLL(0), HLL(1)]

            Where,

            HLL(i) = HLL(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_HLL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hll_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hll_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hlh(self, diff_i, diff_j, diff_k):
        """Builds a HLH tensor

            T_HLH = [HLH(0), HLH(1)]

            Where,

            HLH(i) = HLH(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_HLH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hlh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hlh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hhl(self, diff_i, diff_j, diff_k):
        """Builds a HHL tensor

            T_HHL = [HHL(0), HHL(1)]

            Where,

            HHL(i) = HHL(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_HHL
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hhl_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hhl_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hhh(self, diff_i, diff_j, diff_k):
        """Builds a HHH tensor

            T_HHH = [HHH(0), HHH(1)]

            Where,

            HHH(i) = HHH(i, diff_i, diff_j, diff_k)

            Args:
                diff_i := diff on first basis
                diff_j := diff on second basis
                diff_k := diff on third basis

            Returns:
                tensor T_HHH
        """

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hhh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hhh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_lll_arrays(self):
        """Build and set all the LLL arrays

            Sets:
                LLL_110 = T_LLL(1, 1, 0)
                LLL_200 = T_LLL(1, 0, 1)
        """

        self.LLL_110 = self._build_lll(1, 1, 0)
        self.LLL_200 = self._build_lll(1, 0, 1)

    def _build_llh_arrays(self):
        """Build and set all the LLH arrays

            Sets:
                LLH_110 = T_LLH(1, 1, 0)
                LLH_200 = T_LLH(1, 0, 1)
        """

        self.LLH_110 = self._build_llh(1, 1, 0)
        self.LLH_200 = self._build_llh(1, 0, 1)

    def _build_lhl_arrays(self):
        """Build and set all the LHL arrays

            Sets:
                LHL_110 = T_LHL(1, 1, 0)
                LHL_200 = T_LHL(1, 0, 1)
        """

        self.LHL_110 = self._build_lhl(1, 1, 0)
        self.LHL_200 = self._build_lhl(1, 0, 1)

    def _build_lhh_arrays(self):
        """Build and set all the LHH arrays

            Sets:
                LHH_110 = T_LHH(1, 1, 0)
                LHH_200 = T_LHH(1, 0, 1)
        """

        self.LHH_110 = self._build_lhh(1, 1, 0)
        self.LHH_200 = self._build_lhh(1, 0, 1)

    def _build_hll_arrays(self):
        """Build and set all the HLL arrays

            Sets:
                HLL_110 = T_HLL(1, 1, 0)
                HLL_200 = T_HLL(1, 0, 1)
        """

        self.HLL_110 = self._build_hll(1, 1, 0)
        self.HLL_200 = self._build_hll(1, 0, 1)

    def _build_hlh_arrays(self):
        """Build and set all the HLH arrays

            Sets:
                HLH_110 = T_HLH(1, 1, 0)
                HLH_200 = T_HLH(1, 0, 1)
        """

        self.HLH_110 = self._build_hlh(1, 1, 0)
        self.HLH_200 = self._build_hlh(1, 0, 1)

    def _build_hhl_arrays(self):
        """Build and set all the HHL arrays

            Sets:
                HHL_110 = T_HHL(1, 1, 0)
                HHL_200 = T_HHL(1, 0, 1)
        """

        self.HHL_110 = self._build_hhl(1, 1, 0)
        self.HHL_200 = self._build_hhl(1, 0, 1)

    def _build_hhh_arrays(self):
        """Build and set all the HHH arrays

            Sets:
                HHH_110 = T_HHH(1, 1, 0)
                HHH_200 = T_HHH(1, 0, 1)
        """

        self.HHH_110 = self._build_hhh(1, 1, 0)
        self.HHH_200 = self._build_hhh(1, 0, 1)

    def _build_arrays(self):
        """Builds all the arrays

            Runs:
                _build_lll_arrays
                _build_llh_arrays
                _build_lhl_arrays
                _build_lhh_arrays
                _build_hll_arrays
                _build_hlh_arrays
                _build_hhl_arrays
                _build_hhh_arrays
        """

        self._build_lll_arrays()
        self._build_llh_arrays()
        self._build_lhl_arrays()
        self._build_lhh_arrays()
        self._build_hll_arrays()
        self._build_hlh_arrays()
        self._build_hhl_arrays()
        self._build_hhh_arrays()

    def _build_ll(self):
        """Build all the LL matrices

            Runs:
                _build_ll_00
                _build_ll_01
                _build_ll_02
                _build_ll_10
                _build_ll_11
                _build_ll_12
                _build_ll_20
                _build_ll_21
                _build_ll_22
        """

        self._build_ll_00()
        self._build_ll_01()
        self._build_ll_02()
        self._build_ll_10()
        self._build_ll_11()
        self._build_ll_12()
        self._build_ll_20()
        self._build_ll_21()
        self._build_ll_22()

    def _build_lh(self):
        """Build all the LH matrices

            Runs:
                _build_lh_00
                _build_lh_01
                _build_lh_02
                _build_lh_10
                _build_lh_11
                _build_lh_12
                _build_lh_20
                _build_lh_21
                _build_lh_22
        """

        self._build_lh_00()
        self._build_lh_01()
        self._build_lh_02()
        self._build_lh_10()
        self._build_lh_11()
        self._build_lh_12()
        self._build_lh_20()
        self._build_lh_21()
        self._build_lh_22()

    def _build_hl(self):
        """Build all the HL matrices

            Runs:
                _build_hl_00
                _build_hl_01
                _build_hl_02
                _build_hl_10
                _build_hl_11
                _build_hl_12
                _build_hl_20
                _build_hl_21
                _build_hl_22
        """

        self._build_hl_00()
        self._build_hl_01()
        self._build_hl_02()
        self._build_hl_10()
        self._build_hl_11()
        self._build_hl_12()
        self._build_hl_20()
        self._build_hl_21()
        self._build_hl_22()

    def _build_hh(self):
        """Build all the HH matrices

            Runs:
                _build_hh_00
                _build_hh_01
                _build_hh_02
                _build_hh_10
                _build_hh_11
                _build_hh_12
                _build_hh_20
                _build_hh_21
                _build_hh_22
        """

        self._build_hh_00()
        self._build_hh_01()
        self._build_hh_02()
        self._build_hh_10()
        self._build_hh_11()
        self._build_hh_12()
        self._build_hh_20()
        self._build_hh_21()
        self._build_hh_22()

    def _build(self):
        """Build and set all matrices and tensors

            Runs:
                _build_ll
                _build_lh
                _build_hl
                _build_hh
                _build_arrays
        """

        self._build_ll()
        self._build_lh()
        self._build_hl()
        self._build_hh()

        self._build_arrays()

    def function_l_product(self, f, index, diff):
        """Get the function Lagrange basis product

            f_L = f * L(index, diff)

            Args:
                f     := function for inner product
                index := index of basis
                diff  := derivative on basis

            Returns:
                f_L
        """

        func = self.build_lagrange(index, diff)

        return func * f

    def function_h_product(self, f, index, diff):
        """Get the function Hermite basis product

            f_H = f * H(index, diff)

            Args:
                f     := function for inner product
                index := index of basis
                diff  := derivative on basis

            Returns:
                f_H
        """

        func = self.build_hermite(index, diff)

        return func * f

    def function_l_vector(self, f, diff=0):
        """Get the vector of function-Lagrange evaluations

            F_L(i) = f_L(i, diff)

            Where,

            vec_L = [F_L(0), F_L(1)]

            Args:
                f    := function for inner product
                diff := derivative on basis

            Returns:
                returns vec_L
        """
       
        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_0 = self.function_l_product(f, basis_index0, diff)
        func_1 = self.function_l_product(f, basis_index1, diff)

        point_0 = func_0.integral(self.lower_bound, self.upper_bound)
        point_1 = func_1.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_0], [point_1]])

    def function_h_vector(self, f, diff=0):
        """Get the vector of function-Hermite evaluations

            F_H(i) = f_H(i, diff)

            Where,

            vec_H = [F_H(0), F_H(1)]

            Args:
                f    := function for inner product
                diff := derivative on basis

            Returns:
                returns vec_H
        """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_0 = self.function_h_product(f, basis_index0, diff)
        func_1 = self.function_h_product(f, basis_index1, diff)

        point_0 = func_0.integral(self.lower_bound, self.upper_bound)
        point_1 = func_1.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_0], [point_1]])
