import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin

from source import element


class FiniteElements(object):
    """Class to contain finite elements system

        Attributes:
            _num_points := number of points
            _delta_x    := distance between points

            _elements := list of elements

            A_00 := the 00 FEM matrix
            A_01 := the 01 FEM matrix
            A_02 := the 02 FEM matrix
            A_10 := the 10 FEM matrix
            A_11 := the 11 FEM matrix
            A_12 := the 12 FEM matrix
            A_20 := the 20 FEM matrix
            A_21 := the 21 FEM matrix
            A_22 := the 22 FEM matrix
    """

    def __init__(self, num_points, delta_x):
        self._num_points = num_points
        self._delta_x = delta_x

    @classmethod
    def setup(cls, num_points, delta_x):
        """Setup finite_elements

            Args:
                num_points := number of points
                delta_x    := distance between points

            Returns:
                fully built finite elements
        """

        new_finite = cls(num_points, delta_x)
        new_finite._build()

        return new_finite

    @property
    def num_elements(self):
        """The number of elements

            Returns:
                number of elements
        """

        return self._num_points - 1

    def build_element(self, index):
        """Build an element

            Args:
                index := index of element

            Returns:
                the fully built element with index
        """

        return element.Element.setup(index, self._num_points, self._delta_x)

    def _build_elements(self):
        """Build all the elements

            Runs:
                build for every element upto index num_elements
        """

        self._elements = []

        for index in range(self.num_elements):
            self._elements.append(self.build_element(index))

    def _local_pl_2(self, p, index):
        """Get the local PL array from element

            Args:
                p      := function for PL
                index := index of element

            Returns:
                PL_2 array for element index
        """

        return self._elements[index].pl_2(p)

    def _local_ph_2(self, p, index):
        """Get the local PH array from element

            Args:
                p      := function for PH
                index := index of element

            Returns:
                PH_2 array for element index
        """

        return self._elements[index].ph_2(p)

    def _local_ll_00(self, index):
        """Get the local LL_00 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_00 for element of index
        """

        return self._elements[index].LL_00

    def _local_ll_01(self, index):
        """Get the local LL_01 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_01 for element of index
        """

        return self._elements[index].LL_01

    def _local_ll_02(self, index):
        """Get the local LL_02 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_02 for element of index
        """

        return self._elements[index].LL_02

    def _local_ll_10(self, index):
        """Get the local LL_10 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_10 for element of index
        """

        return self._elements[index].LL_10

    def _local_ll_11(self, index):
        """Get the local LL_11 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_11 for element of index
        """

        return self._elements[index].LL_11

    def _local_ll_12(self, index):
        """Get the local LL_12 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_12 for element of index
        """

        return self._elements[index].LL_12

    def _local_ll_20(self, index):
        """Get the local LL_20 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_20 for element of index
        """

        return self._elements[index].LL_20

    def _local_ll_21(self, index):
        """Get the local LL_21 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_21 for element of index
        """

        return self._elements[index].LL_21

    def _local_ll_22(self, index):
        """Get the local LL_22 matrix from element index

            Args:
                index := index of element

            Returns:
                LL_22 for element of index
        """

        return self._elements[index].LL_22

    def _local_lh_00(self, index):
        """Get the local LH_00 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_00 for element of index
        """

        return self._elements[index].LH_00

    def _local_lh_01(self, index):
        """Get the local LH_01 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_01 for element of index
        """

        return self._elements[index].LH_01

    def _local_lh_02(self, index):
        """Get the local LH_02 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_02 for element of index
        """

        return self._elements[index].LH_02

    def _local_lh_10(self, index):
        """Get the local LH_10 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_10 for element of index
        """

        return self._elements[index].LH_10

    def _local_lh_11(self, index):
        """Get the local LH_11 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_11 for element of index
        """

        return self._elements[index].LH_11

    def _local_lh_12(self, index):
        """Get the local LH_12 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_12 for element of index
        """

        return self._elements[index].LH_12

    def _local_lh_20(self, index):
        """Get the local LH_20 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_20 for element of index
        """

        return self._elements[index].LH_20

    def _local_lh_21(self, index):
        """Get the local LH_21 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_21 for element of index
        """

        return self._elements[index].LH_21

    def _local_lh_22(self, index):
        """Get the local LH_22 matrix from element index

            Args:
                index := index of element

            Returns:
                LH_22 for element of index
        """

        return self._elements[index].LH_22

    def _local_hl_00(self, index):
        """Get the local HL_00 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_00 for element of index
        """

        return self._elements[index].HL_00

    def _local_hl_01(self, index):
        """Get the local HL_01 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_01 for element of index
        """

        return self._elements[index].HL_01

    def _local_hl_02(self, index):
        """Get the local HL_02 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_02 for element of index
        """

        return self._elements[index].HL_02

    def _local_hl_10(self, index):
        """Get the local HL_10 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_10 for element of index
        """

        return self._elements[index].HL_10

    def _local_hl_11(self, index):
        """Get the local HL_11 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_11 for element of index
        """

        return self._elements[index].HL_11

    def _local_hl_12(self, index):
        """Get the local HL_12 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_12 for element of index
        """

        return self._elements[index].HL_12

    def _local_hl_20(self, index):
        """Get the local HL_20 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_20 for element of index
        """

        return self._elements[index].HL_20

    def _local_hl_21(self, index):
        """Get the local HL_21 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_21 for element of index
        """

        return self._elements[index].HL_21

    def _local_hl_22(self, index):
        """Get the local HL_22 matrix from element index

            Args:
                index := index of element

            Returns:
                HL_22 for element of index
        """

        return self._elements[index].HL_22

    def _local_hh_00(self, index):
        """Get the local HH_00 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_00 for element of index
        """

        return self._elements[index].HH_00

    def _local_hh_01(self, index):
        """Get the local HH_01 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_01 for element of index
        """

        return self._elements[index].HH_01

    def _local_hh_02(self, index):
        """Get the local HH_02 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_02 for element of index
        """

        return self._elements[index].HH_02

    def _local_hh_10(self, index):
        """Get the local HH_10 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_10 for element of index
        """

        return self._elements[index].HH_10

    def _local_hh_11(self, index):
        """Get the local HH_11 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_11 for element of index
        """

        return self._elements[index].HH_11

    def _local_hh_12(self, index):
        """Get the local HH_12 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_12 for element of index
        """

        return self._elements[index].HH_12

    def _local_hh_20(self, index):
        """Get the local HH_20 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_20 for element of index
        """

        return self._elements[index].HH_20

    def _local_hh_21(self, index):
        """Get the local HH_21 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_21 for element of index
        """

        return self._elements[index].HH_21

    def _local_hh_22(self, index):
        """Get the local HH_22 matrix from element index

            Args:
                index := index of element

            Returns:
                HH_22 for element of index
        """

        return self._elements[index].HH_22

    def _local_lll_110(self, index):
        """Gets the local LLL_110 array from element of index

            Args:
                index := index of element

            Returns:
                LLL_110 for element of index
        """

        return self._elements[index].LLL_110

    def _local_lll_200(self, index):
        """Gets the local LLL_200 array from element of index

            Args:
                index := index of element

            Returns:
                LLL_200 for element of index
        """

        return self._elements[index].LLL_200

    def _local_llh_110(self, index):
        """Gets the local LLH_110 array from element of index

            Args:
                index := index of element

            Returns:
                LLH_110 for element of index
        """

        return self._elements[index].LLH_110

    def _local_llh_200(self, index):
        """Gets the local LLH_200 array from element of index

            Args:
                index := index of element

            Returns:
                LLH_200 for element of index
        """

        return self._elements[index].LLH_200

    def _local_lhl_110(self, index):
        """Gets the local LHL_110 array from element of index

            Args:
                index := index of element

            Returns:
                LHL_110 for element of index
        """

        return self._elements[index].LHL_110

    def _local_lhl_200(self, index):
        """Gets the local LHL_200 array from element of index

            Args:
                index := index of element

            Returns:
                LHL_200 for element of index
        """

        return self._elements[index].LHL_200

    def _local_lhh_110(self, index):
        """Gets the local LLL_110 array from element of index

            Args:
                index := index of element

            Returns:
                LLL_110 for element of index
        """

        return self._elements[index].LHH_110

    def _local_lhh_200(self, index):
        """Gets the local LLL_200 array from element of index

            Args:
                index := index of element

            Returns:
                LLL_200 for element of index
        """

        return self._elements[index].LHH_200

    def _local_hll_110(self, index):
        """Gets the local HLL_110 array from element of index

            Args:
                index := index of element

            Returns:
                HLL_110 for element of index
        """

        return self._elements[index].HLL_110

    def _local_hll_200(self, index):
        """Gets the local HLL_200 array from element of index

            Args:
                index := index of element

            Returns:
                HLL_200 for element of index
        """

        return self._elements[index].HLL_200

    def _local_hlh_110(self, index):
        """Gets the local HLH_110 array from element of index

            Args:
                index := index of element

            Returns:
                HLH_110 for element of index
        """

        return self._elements[index].HLH_110

    def _local_hlh_200(self, index):
        """Gets the local HLH_200 array from element of index

            Args:
                index := index of element

            Returns:
                HLH_200 for element of index
        """

        return self._elements[index].HLH_200

    def _local_hhl_110(self, index):
        """Gets the local HHL_110 array from element of index

            Args:
                index := index of element

            Returns:
                HHL_110 for element of index
        """

        return self._elements[index].HHL_110

    def _local_hhl_200(self, index):
        """Gets the local HHL_200 array from element of index

            Args:
                index := index of element

            Returns:
                HHL_200 for element of index
        """

        return self._elements[index].HHL_200

    def _local_hhh_110(self, index):
        """Gets the local HHH_110 array from element of index

            Args:
                index := index of element

            Returns:
                HHH_110 for element of index
        """

        return self._elements[index].HHH_110

    def _local_hhh_200(self, index):
        """Gets the local HHH_200 array from element of index

            Args:
                index := index of element

            Returns:
                HHH_200 for element of index
        """

        return self._elements[index].HHH_200

    def _local_function_l(self, f, index, diff=0):
        """Get the local Lagrange projection of f

            Args:
                f     := function for projection
                index := index of elements
                diff  := derivative of basis function

            Returns:
                projection array
        """

        return self._elements[index].function_l_vector(f, diff=diff)

    def _local_function_h(self, f, index, diff=0):
        """Get the local Hermite projection of f

            Args:
                f     := function for projection
                index := index of elements
                diff  := derivative of basis function

            Returns:
                projection array
        """

        return self._elements[index].function_h_vector(f, diff=diff)

    def _init_block(self):
        """Create an initial block matrix

            Returns:
                empty sparse matrix
        """

        return sp.lil_matrix((self._num_points, self._num_points),
                             dtype=np.float64)

    def _build_block(self, local_matrix):
        """Build one of the block matrices using local_matrix

            Args:
                local_matrix := function to construct local (to element) matrix

            Returns:
                the block matrix
        """

        block = self._init_block()
        for index in range(self.num_elements):
            local_block = local_matrix(index)
            block[index:(index + 2), index:(index + 2)] += local_block

        return block

    def _build_ll_00(self):
        """Build the LL_00 block matrix

            Returns:
                block LL_00 matrix
        """

        return self._build_block(self._local_ll_00)

    def _build_ll_01(self):
        """Build the LL_01 block matrix

            Returns:
                block LL_01 matrix
        """

        return self._build_block(self._local_ll_01)

    def _build_ll_02(self):
        """Build the LL_02 block matrix

            Returns:
                block LL_02 matrix
        """

        return self._build_block(self._local_ll_02)

    def _build_ll_10(self):
        """Build the LL_10 block matrix

            Returns:
                block LL_10 matrix
        """

        return self._build_block(self._local_ll_10)

    def _build_ll_11(self):
        """Build the LL_11 block matrix

            Returns:
                block LL_11 matrix
        """

        return self._build_block(self._local_ll_11)

    def _build_ll_12(self):
        """Build the LL_12 block matrix

            Returns:
                block LL_12 matrix
        """

        return self._build_block(self._local_ll_12)

    def _build_ll_20(self):
        """Build the LL_20 block matrix

            Returns:
                block LL_20 matrix
        """

        return self._build_block(self._local_ll_20)

    def _build_ll_21(self):
        """Build the LL_21 block matrix

            Returns:
                block LL_21 matrix
        """

        return self._build_block(self._local_ll_21)

    def _build_ll_22(self):
        """Build the LL_22 block matrix

            Returns:
                block LL_22 matrix
        """

        return self._build_block(self._local_ll_22)

    def _build_lh_00(self):
        """Build the LH_00 block matrix

            Returns:
                block LH_00 matrix
        """

        return self._build_block(self._local_lh_00)

    def _build_lh_01(self):
        """Build the LH_01 block matrix

            Returns:
                block LH_01 matrix
        """

        return self._build_block(self._local_lh_01)

    def _build_lh_02(self):
        """Build the LH_02 block matrix

            Returns:
                block LH_02 matrix
        """

        return self._build_block(self._local_lh_02)

    def _build_lh_10(self):
        """Build the LH_10 block matrix

            Returns:
                block LH_10 matrix
        """

        return self._build_block(self._local_lh_10)

    def _build_lh_11(self):
        """Build the LH_11 block matrix

            Returns:
                block LH_11 matrix
        """

        return self._build_block(self._local_lh_11)

    def _build_lh_12(self):
        """Build the LH_12 block matrix

            Returns:
                block LH_12 matrix
        """

        return self._build_block(self._local_lh_12)

    def _build_lh_20(self):
        """Build the LH_20 block matrix

            Returns:
                block LH_20 matrix
        """

        return self._build_block(self._local_lh_20)

    def _build_lh_21(self):
        """Build the LH_21 block matrix

            Returns:
                block LH_21 matrix
        """

        return self._build_block(self._local_lh_21)

    def _build_lh_22(self):
        """Build the LH_22 block matrix

            Returns:
                block LH_22 matrix
        """

        return self._build_block(self._local_lh_22)

    def _build_hl_00(self):
        """Build the HL_00 block matrix

            Returns:
                block HL_00 matrix
        """

        return self._build_block(self._local_hl_00)

    def _build_hl_01(self):
        """Build the HL_01 block matrix

            Returns:
                block HL_01 matrix
        """

        return self._build_block(self._local_hl_01)

    def _build_hl_02(self):
        """Build the HL_02 block matrix

            Returns:
                block HL_02 matrix
        """

        return self._build_block(self._local_hl_02)

    def _build_hl_10(self):
        """Build the HL_10 block matrix

            Returns:
                block HL_10 matrix
        """

        return self._build_block(self._local_hl_10)

    def _build_hl_11(self):
        """Build the HL_11 block matrix

            Returns:
                block HL_11 matrix
        """

        return self._build_block(self._local_hl_11)

    def _build_hl_12(self):
        """Build the HL_12 block matrix

            Returns:
                block HL_12 matrix
        """

        return self._build_block(self._local_hl_12)

    def _build_hl_20(self):
        """Build the HL_20 block matrix

            Returns:
                block HL_20 matrix
        """

        return self._build_block(self._local_hl_20)

    def _build_hl_21(self):
        """Build the HL_21 block matrix

            Returns:
                block HL_21 matrix
        """

        return self._build_block(self._local_hl_21)

    def _build_hl_22(self):
        """Build the HL_22 block matrix

            Returns:
                block HL_22 matrix
        """

        return self._build_block(self._local_hl_22)

    def _build_hh_00(self):
        """Build the HH_00 block matrix

            Returns:
                block HH_00 matrix
        """

        return self._build_block(self._local_hh_00)

    def _build_hh_01(self):
        """Build the HH_01 block matrix

            Returns:
                block HH_01 matrix
        """

        return self._build_block(self._local_hh_01)

    def _build_hh_02(self):
        """Build the HH_02 block matrix

            Returns:
                block HH_02 matrix
        """

        return self._build_block(self._local_hh_02)

    def _build_hh_10(self):
        """Build the HH_10 block matrix

            Returns:
                block HH_10 matrix
        """

        return self._build_block(self._local_hh_10)

    def _build_hh_11(self):
        """Build the HH_11 block matrix

            Returns:
                block HH_11 matrix
        """

        return self._build_block(self._local_hh_11)

    def _build_hh_12(self):
        """Build the HH_12 block matrix

            Returns:
                block HH_12 matrix
        """

        return self._build_block(self._local_hh_12)

    def _build_hh_20(self):
        """Build the HH_20 block matrix

            Returns:
                block HH_20 matrix
        """

        return self._build_block(self._local_hh_20)

    def _build_hh_21(self):
        """Build the HH_21 block matrix

            Returns:
                block HH_21 matrix
        """

        return self._build_block(self._local_hh_21)

    def _build_hh_22(self):
        """Build the HH_22 block matrix

            Returns:
                block HH_22 matrix
        """

        return self._build_block(self._local_hh_22)

    def _init_array(self):
        """Builds initial block array

            Returns:
                empty sparse tensor array
        """

        array_list = []

        for _ in range(self._num_points):
            array_list.append(self._init_block())

        return array_list

    def _build_block_array(self, local_array):
        """Builds one of the block arrays

            Args:
                local_array := function to construct local (to element) array

            Returns:
                one of the block arrays
        """

        block = self._init_array()

        for index in range(self.num_elements):
            local_block_arrays = local_array(index)
            for l_index, l_block in enumerate(local_block_arrays):
                block[index + l_index][index:(index + 2), index:(index + 2)] =\
                    l_block

        return block

    def _build_local_block_array_lll_110(self):
        """Builds the local LLL 110 block array

            Returns:
                block LLL_110 array
        """

        return self._build_block_array(self._local_lll_110)

    def _build_local_block_array_lll_200(self):
        """Builds the local LLL 200 block array

            Returns:
                block LLL_200 array
        """

        return self._build_block_array(self._local_lll_200)

    def _build_local_block_array_llh_110(self):
        """Builds the local LLH 110 block array

            Returns:
                block LLH_110 array
        """

        return self._build_block_array(self._local_llh_110)

    def _build_local_block_array_llh_200(self):
        """Builds the local LLH 200 block array

            Returns:
                block LLH_200 array
        """

        return self._build_block_array(self._local_llh_200)

    def _build_local_block_array_lhl_110(self):
        """Builds the local LHL 110 block array

            Returns:
                block LHL_110 array
        """

        return self._build_block_array(self._local_lhl_110)

    def _build_local_block_array_lhl_200(self):
        """Builds the local LHL 200 block array

            Returns:
                block LHL_200 array
        """

        return self._build_block_array(self._local_lhl_200)

    def _build_local_block_array_lhh_110(self):
        """Builds the local LHH 110 block array

            Returns:
                block LHH_110 array
        """

        return self._build_block_array(self._local_lhh_110)

    def _build_local_block_array_lhh_200(self):
        """Builds the local LHH 200 block array

            Returns:
                block LHH_200 array
        """

        return self._build_block_array(self._local_lhh_200)

    def _build_local_block_array_hll_110(self):
        """Builds the local HLL 110 block array

            Returns:
                block HLL_110 array
        """

        return self._build_block_array(self._local_hll_110)

    def _build_local_block_array_hll_200(self):
        """Builds the local HLL 200 block array

            Returns:
                block HLL_200 array
        """

        return self._build_block_array(self._local_hll_200)

    def _build_local_block_array_hlh_110(self):
        """Builds the local HLH 110 block array

            Returns:
                block HLH_110 array
        """

        return self._build_block_array(self._local_hlh_110)

    def _build_local_block_array_hlh_200(self):
        """Builds the local HLH 200 block array

            Returns:
                block HLH_200 array
        """

        return self._build_block_array(self._local_hlh_200)

    def _build_local_block_array_hhl_110(self):
        """Builds the local HHL 110 block array

            Returns:
                block HHL_110 array
        """

        return self._build_block_array(self._local_hhl_110)

    def _build_local_block_array_hhl_200(self):
        """Builds the local HHL 200 block array

            Returns:
                block HHL_200 array
        """

        return self._build_block_array(self._local_hhl_200)

    def _build_local_block_array_hhh_110(self):
        """Builds the local HHH 110 block array

            Returns:
                block HHH_110 array
        """

        return self._build_block_array(self._local_hhh_110)

    def _build_local_block_array_hhh_200(self):
        """Builds the local HHH 200 block array

            Returns:
                block HHH_200 array
        """

        return self._build_block_array(self._local_hhh_200)

    def _init_vec_block(self):
        """Create an initial vector block

            Returns:
                initial block vector
        """

        return np.zeros((self._num_points, 1))

    def _build_vec_block(self, f, local_vec, diff=0):
        """Build the vector block

            Args:
                f         := function for projection
                local_vec := function to create the local vector
                diff      := derivative of basis function

            Returns:
                the block vector
        """

        block = self._init_vec_block()
        for index in range(self.num_elements):
            local_block = local_vec(f, index, diff=diff)
            block[index:(index + 2)] += local_block

        return block

    def _build_vec_l(self, f, diff=0):
        """Build the local vec Lagrange for f

            Args:
                f    := function for projection
                diff := derivative of basis function

            Returns:
                the Lagrange block vector
        """

        return self._build_vec_block(f, self._local_function_l, diff=diff)

    def _build_vec_h(self, f, diff=0):
        """Build the local vec Hermite for f

            Args:
                f    := function for projection
                diff := derivative of basis function

            Returns:
                the Hermite block vector
        """

        return self._build_vec_block(f, self._local_function_h, diff=diff)

    @staticmethod
    def _build_matrix(ll, lh, hl, hh):
        """Build the block matrix

            Args:
                ll := Lagrange-Lagrange matrix
                lh := Lagrange-Hermite matrix
                hl := Hermite-Lagrange matrix
                hh := Hermite-Hermite matrix

            Returns:
                the assembled matrix of blocks
        """

        return sp.bmat([[ll, lh], [hl, hh]])

    def _build_00(self):
        """Build the A_00 matrix

            Sets:
                A_00 matrix
        """

        ll = self._build_ll_00()
        lh = self._build_lh_00()
        hl = self._build_hl_00()
        hh = self._build_hh_00()

        self.A_00 = self._build_matrix(ll, lh, hl, hh)

    def _build_01(self):
        """Build the A_01 matrix

            Sets:
                A_01 matrix
        """

        ll = self._build_ll_01()
        lh = self._build_lh_01()
        hl = self._build_hl_01()
        hh = self._build_hh_01()

        self.A_01 = self._build_matrix(ll, lh, hl, hh)

    def _build_02(self):
        """Build the A_02 matrix

            Sets:
                A_02 matrix
        """

        ll = self._build_ll_02()
        lh = self._build_lh_02()
        hl = self._build_hl_02()
        hh = self._build_hh_02()

        self.A_02 = self._build_matrix(ll, lh, hl, hh)

    def _build_10(self):
        """Build the A_10 matrix

            Sets:
                A_10 matrix
        """

        ll = self._build_ll_10()
        lh = self._build_lh_10()
        hl = self._build_hl_10()
        hh = self._build_hh_10()

        self.A_10 = self._build_matrix(ll, lh, hl, hh)

    def _build_11(self):
        """Build the A_11 matrix

            Sets:
                A_11 matrix
        """

        ll = self._build_ll_11()
        lh = self._build_lh_11()
        hl = self._build_hl_11()
        hh = self._build_hh_11()

        self.A_11 = self._build_matrix(ll, lh, hl, hh)

    def _build_12(self):
        """Build the A_12 matrix

            Sets:
                A_12 matrix
        """

        ll = self._build_ll_12()
        lh = self._build_lh_12()
        hl = self._build_hl_12()
        hh = self._build_hh_12()

        self.A_12 = self._build_matrix(ll, lh, hl, hh)

    def _build_20(self):
        """Build the A_20 matrix

            Sets:
                A_20 matrix
        """

        ll = self._build_ll_20()
        lh = self._build_lh_20()
        hl = self._build_hl_20()
        hh = self._build_hh_20()

        self.A_20 = self._build_matrix(ll, lh, hl, hh)

    def _build_21(self):
        """Build the A_21 matrix

            Sets:
                A_21 matrix
        """

        ll = self._build_ll_21()
        lh = self._build_lh_21()
        hl = self._build_hl_21()
        hh = self._build_hh_21()

        self.A_21 = self._build_matrix(ll, lh, hl, hh)

    def _build_22(self):
        """Build the A_22 matrix

            Sets:
                A_22 matrix
        """

        ll = self._build_ll_22()
        lh = self._build_lh_22()
        hl = self._build_hl_22()
        hh = self._build_hh_22()

        self.A_22 = self._build_matrix(ll, lh, hl, hh)

    def _build_a_matrices(self):
        """Build the A matrices

            Runs:
                _build_00
                _build_01
                _build_02
                _build_10
                _build_11
                _build_12
                _build_20
                _build_21
                _build_22
        """

        self._build_00()
        self._build_01()
        self._build_02()
        self._build_10()
        self._build_11()
        self._build_12()
        self._build_20()
        self._build_21()
        self._build_22()

    def _block_assembler_array(self, ll, lh, hl, hh):
        """Assembles a array from blocks

            Args:
                ll := the ll block array
                lh := the lh block array
                hl := the hl block array
                hh := the hh block array

            Returns:
                the full array
        """

        array_list = []

        for index in range(self._num_points):
            block = sp.bmat([[ll[index], lh[index]],
                             [hl[index], hh[index]]])
            array_list.append(block)

        return array_list

    def _build_block_array_l_110(self):
        """Builds the big block array for Lagrange 110

            Returns:
                Lagrange A_110
        """

        ll = self._build_local_block_array_lll_110()
        lh = self._build_local_block_array_lhl_110()
        hl = self._build_local_block_array_hll_110()
        hh = self._build_local_block_array_hhl_110()

        return self._block_assembler_array(ll, lh, hl, hh)

    def _build_block_array_l_200(self):
        """Builds the big block array for Lagrange 200

            Returns:
                Lagrange A_200
        """

        ll = self._build_local_block_array_lll_200()
        lh = self._build_local_block_array_lhl_200()
        hl = self._build_local_block_array_hll_200()
        hh = self._build_local_block_array_hhl_200()

        return self._block_assembler_array(ll, lh, hl, hh)

    def _build_block_array_h_110(self):
        """Builds the big block array for Hermite 110

            Returns:
                Hermite A_110
        """

        ll = self._build_local_block_array_llh_110()
        lh = self._build_local_block_array_lhh_110()
        hl = self._build_local_block_array_hlh_110()
        hh = self._build_local_block_array_hhh_110()

        return self._block_assembler_array(ll, lh, hl, hh)

    def _build_block_array_h_200(self):
        """Builds the big block array for Hermite 200

            Returns:
                Hermite A_200
        """

        ll = self._build_local_block_array_llh_200()
        lh = self._build_local_block_array_lhh_200()
        hl = self._build_local_block_array_hlh_200()
        hh = self._build_local_block_array_hhh_200()

        return self._block_assembler_array(ll, lh, hl, hh)

    def _build_array_110(self):
        """Builds the collection of all 110 arrays

            Sets:
                A_110
        """

        l_array = self._build_block_array_l_110()
        h_array = self._build_block_array_h_110()

        self.A_110 = l_array + h_array

    def _build_array_200(self):
        """Builds the collection of all 200 arrays

            Sets:
                A_200
        """

        l_array = self._build_block_array_l_200()
        h_array = self._build_block_array_h_200()

        self.A_200 = l_array + h_array

    def build_vec(self, f, diff=0):
        """Build the vector for function f

            Args:
                f    := function for projection
                diff := derivative of basis function

            Returns:
                the block vector
        """

        l_vec = self._build_vec_l(f, diff=diff)
        h_vec = self._build_vec_h(f, diff=diff)

        return np.concatenate((l_vec, h_vec))

    def _build(self):
        """Build all sub components

            Runs:
                _build_elements
                _build_a_matrices
                _build_array_110
                _build_array_200
        """

        self._build_elements()
        self._build_a_matrices()
        self._build_array_110()
        self._build_array_200()

    def l2_project(self, f, identify):
        """Find l2 projection of f

            P = A_00^-1 * (f, b_j)_j=0^num_points-1

            Args:
                f        := function to project
                identify := decide if we identify

            Returns:
                Projection of f
        """

        b = self.build_vec(f)
        M = self.A_00.tocsc()
        Mtemp = M.copy()

        if identify:
            for k in range(M.shape[0]):
                Mtemp[k, 0] = 0.0
                Mtemp[0, k] = 0.0
                Mtemp[k, self._num_points] = 0.0
                Mtemp[self._num_points, k] = 0.0
            Mtemp[0, 0] = 1.0
            Mtemp[self._num_points, self._num_points] = 1.0

        return lin.spsolve(Mtemp, b)

    def h1_project(self, f):
        """Find h1 projection of f

            P = (A_00 - A_01)^-1 * (f, b_j)_j=0^num_points-1

            Args:
                f := function to project

            Returns:
                Projection of f
        """

        b = self.build_vec(f)
        M = self.A_00.tocsc() - self.A_01.tocsc()

        return lin.spsolve(M, b)

    def h2_project(self, f):
        """Find h1 projection of f

            P = (A_00 - A_01 + A_02)^-1 * (f, b_j)_j=0^num_points-1

            Args:
            f := function to project

            Returns:
            Projection of f
        """

        b = self.build_vec(f)
        M = self.A_00.tocsc() - self.A_01.tocsc() + self.A_02.tocsc()

        return lin.spsolve(M, b)

    def l2_norm(self, u):
        """Find the l2 norm of vec

            Args:
                u := function to take norm of

            Returns:
                norm
        """

        v = self.A_00 * u
        value = np.dot(u, v)

        return value * 0.5
