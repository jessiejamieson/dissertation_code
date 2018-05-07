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
        """ Setup the element """

        new_element = cls(index, num_points, delta_x)
        new_element._build()

        return new_element

    @property
    def lower_bound(self):
        """ Get the lower bound of interval """

        return self._index * self._delta_x

    @property
    def upper_bound(self):
        """ Get the upper bound of interval """

        return (self._index + 1) * self._delta_x

    @property
    def basis_index(self):
        """ Get the indices for nonzero basis functions on element """

        return [self._index, (self._index + 1)]

    def build_lagrange(self, index, derivative):
        """ Build a lagrange basis function """

        return basis.Lagrange(index, derivative,
                              self._num_points, self._delta_x)

    def build_hermite(self, index, derivative):
        """ Build a hermite basis function """

        return basis.Hermite(index, derivative,
                             self._num_points, self._delta_x)

    def func_pl(self, p, index0, diff0):
        """ find the P-lagrange product"""

        function1 = p
        function0 = self.build_lagrange(index0, diff0)

        return function0 * function1

    def func_ph(self, p, index0, diff0):
        """ find the P-hermite product"""

        function1 = p
        function0 = self.build_hermite(index0, diff0)

        return function0 * function1

    def func_ll(self, index0, index1, diff0, diff1):
        """ Find the Lagrange-Lagrange product """

        function0 = self.build_lagrange(index0, diff0)
        function1 = self.build_lagrange(index1, diff1)

        return function0 * function1

    def func_lh(self, index0, index1, diff0, diff1):
        """ Find the Lagrange-Hermite product """

        function0 = self.build_lagrange(index0, diff0)
        function1 = self.build_hermite(index1, diff1)

        return function0 * function1

    def func_hl(self, index0, index1, diff0, diff1):
        """ Find the Hermite-Lagrange product """

        function0 = self.build_hermite(index0, diff0)
        function1 = self.build_lagrange(index1, diff1)

        return function0 * function1

    def func_hh(self, index0, index1, diff0, diff1):
        """ Find the Hermite-Hermite product """

        function0 = self.build_hermite(index0, diff0)
        function1 = self.build_hermite(index1, diff1)

        return function0 * function1

    def func_lll(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the Lagrange-Lagrange-Lagrange product"""

        function0 = self.func_ll(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_llh(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the Lagrange-Lagrange-Hermite product"""

        function0 = self.func_ll(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def func_lhl(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the La-He-la product"""

        function0 = self.func_lh(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_lhh(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the La-He-He product"""

        function0 = self.func_lh(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def func_hll(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the Hermite-Lagrange-Lagrange product"""

        function0 = self.func_hl(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_hlh(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the Hermite-Lagrange-Hermite product"""

        function0 = self.func_hl(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def func_hhl(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the Hermite-Hermite-Lagrange product"""

        function0 = self.func_hh(index0, index1, diff0, diff1)
        function1 = self.build_lagrange(index2, diff2)

        return function0 * function1

    def func_hhh(self, index0, index1, index2, diff0, diff1, diff2):
        """finds the Hermite-Hermite-Hermite product"""

        function0 = self.func_hh(index0, index1, diff0, diff1)
        function1 = self.build_hermite(index2, diff2)

        return function0 * function1

    def pl_2(self, p):
        """Build the pl_2 matrix"""

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func0 = self.func_pl(p, basis_index0, 2)
        func1 = self.func_pl(p, basis_index1, 2)

        point_0 = func0.integral(self.lower_bound, self.upper_bound)
        point_1 = func1.integral(self.lower_bound, self.upper_bound)

        return np.array([point_0, point_1])

    def ph_2(self, p):
        """Build the ph_2 matrix"""

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func0 = self.func_ph(p, basis_index0, 2)
        func1 = self.func_ph(p, basis_index1, 2)

        point_0 = func0.integral(self.lower_bound, self.upper_bound)
        point_1 = func1.integral(self.lower_bound, self.upper_bound)

        return np.array([point_0, point_1])

    def _build_ll_00(self):
        """ Build the ll_00 matrix """

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
        """ Build the ll_01 matrix """

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
        """ Build the ll_02 matrix """

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
        """ Build the ll_10 matrix """

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
        """ Build the ll_11 matrix """

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
        """ Build the ll_12 matrix """

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
        """ Build the ll_20 matrix """

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
        """ Build the ll_21 matrix """

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
        """ Build the ll_22 matrix """

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
        """ Build the lh_00 matrix """

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
        """ Build the lh_01 matrix """

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
        """ Build the lh_02 matrix """

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
        """ Build the lh_10 matrix """

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
        """ Build the lh_11 matrix """

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
        """ Build the lh_12 matrix """

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
        """ Build the lh_20 matrix """

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
        """ Build the lh_21 matrix """

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
        """ Build the lh_22 matrix """

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
        """ Build the hl_00 matrix """

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
        """ Build the hl_01 matrix """

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
        """ Build the hl_02 matrix """

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
        """ Build the hl_10 matrix """

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
        """ Build the hl_11 matrix """

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
        """ Build the hl_12 matrix """

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
        """ Build the hl_20 matrix """

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
        """ Build the hl_21 matrix """

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
        """ Build the hl_22 matrix """

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
        """ Build the HH_00 matrix """

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
        """ Build the HH_01 matrix """

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
        """ Build the HH_02 matrix """

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
        """ Build the HH_10 matrix """

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
        """ Build the HH_11 matrix """

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
        """ Build the HH_12 matrix """

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
        """ Build the HH_20 matrix """

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
        """ Build the HH_21 matrix """

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
        """ Build the HH_22 matrix """

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
        """builds the LLL matrix for a fixed k"""

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
        """builds the LLH matrix for a fixed k"""

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
        """builds the LHL matrix for a fixed k"""

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
        """builds the LHH matrix for a fixed k"""

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
        """builds the HLL matrix for a fixed k"""

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
        """builds the HLH matrix for a fixed k"""

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
        """builds the HHL matrix for a fixed k"""

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
        """builds the HHH matrix for a fixed k"""

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
        """builds the LLL tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_lll_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_lll_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_llh(self, diff_i, diff_j, diff_k):
        """builds the LLH tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_llh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_llh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_lhl(self, diff_i, diff_j, diff_k):
        """builds the LHL tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_lhl_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_lhl_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_lhh(self, diff_i, diff_j, diff_k):
        """builds the LHH tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_lhh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_lhh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hll(self, diff_i, diff_j, diff_k):
        """builds the HLL tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hll_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hll_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hlh(self, diff_i, diff_j, diff_k):
        """builds the HLH tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hlh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hlh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hhl(self, diff_i, diff_j, diff_k):
        """builds the HHL tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hhl_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hhl_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_hhh(self, diff_i, diff_j, diff_k):
        """builds the HHH tensor"""

        index0 = self.basis_index[0]
        index1 = self.basis_index[1]

        matrix0 = self._build_hhh_matrix(index0, diff_i, diff_j, diff_k)
        matrix1 = self._build_hhh_matrix(index1, diff_i, diff_j, diff_k)

        return [matrix0, matrix1]

    def _build_lll_arrays(self):
        """sets all the LLL arrays"""

        self.LLL_110 = self._build_lll(1, 1, 0)
        self.LLL_200 = self._build_lll(1, 0, 1)

    def _build_llh_arrays(self):
        """sets all the LLH arrays"""

        self.LLH_110 = self._build_llh(1, 1, 0)
        self.LLH_200 = self._build_llh(1, 0, 1)

    def _build_lhl_arrays(self):
        """sets all the LHL arrays"""

        self.LHL_110 = self._build_lhl(1, 1, 0)
        self.LHL_200 = self._build_lhl(1, 0, 1)

    def _build_lhh_arrays(self):
        """sets all the LHH arrays"""

        self.LHH_110 = self._build_lhh(1, 1, 0)
        self.LHH_200 = self._build_lhh(1, 0, 1)

    def _build_hll_arrays(self):
        """sets all the HLL arrays"""

        self.HLL_110 = self._build_hll(1, 1, 0)
        self.HLL_200 = self._build_hll(1, 0, 1)

    def _build_hlh_arrays(self):
        """sets all the HLH arrays"""

        self.HLH_110 = self._build_hlh(1, 1, 0)
        self.HLH_200 = self._build_hlh(1, 0, 1)

    def _build_hhl_arrays(self):
        """sets all the HHL arrays"""

        self.HHL_110 = self._build_hhl(1, 1, 0)
        self.HHL_200 = self._build_hhl(1, 0, 1)

    def _build_hhh_arrays(self):
        """sets all the HHH arrays"""

        self.HHH_110 = self._build_hhh(1, 1, 0)
        self.HHH_200 = self._build_hhh(1, 0, 1)

    def _build_arrays(self):
        """builds all the arrays"""

        self._build_lll_arrays()
        self._build_llh_arrays()
        self._build_lhl_arrays()
        self._build_lhh_arrays()
        self._build_hll_arrays()
        self._build_hlh_arrays()
        self._build_hhl_arrays()
        self._build_hhh_arrays()

    def _build_ll(self):
        """Build all the LL matrices"""

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
        """Build all the LH matrices"""

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
        """Build all the HL matrices"""

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
        """Build all the HH matrices"""

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
        """ Build and set all matrices """

        self._build_ll()
        self._build_lh()
        self._build_hl()
        self._build_hh()

        self._build_arrays()

    def function_l_product(self, f, index, diff):
        """ get the function L basis product """

        func = self.build_lagrange(index, diff)

        return func * f

    def function_h_product(self, f, index, diff):
        """ get the function H basis product """

        func = self.build_hermite(index, diff)

        return func * f

    def function_l_vector(self, f, diff = 0):
        """ Get the vector of function evaluations """
       
        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_0 = self.function_l_product(f, basis_index0, diff)
        func_1 = self.function_l_product(f, basis_index1, diff)

        point_0 = func_0.integral(self.lower_bound, self.upper_bound)
        point_1 = func_1.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_0], [point_1]])

    def function_h_vector(self, f, diff = 0):
        """ Get the vector of function evaluations """

        basis_index0 = self.basis_index[0]
        basis_index1 = self.basis_index[1]

        func_0 = self.function_h_product(f, basis_index0, diff)
        func_1 = self.function_h_product(f, basis_index1, diff)

        point_0 = func_0.integral(self.lower_bound, self.upper_bound)
        point_1 = func_1.integral(self.lower_bound, self.upper_bound)

        return np.array([[point_0], [point_1]])


