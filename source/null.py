import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin

from source import finite_elements as fem
from source import solver


class NullControl(object):
    """Class to contain the null control system

        Attributes:
            _num_points := the number of space points (includes endpoints)
            _delta_x    := distance between space points
            _num_steps  := number of time steps to take
            _delta_t    := amount of time to take in one timestep

            _gamma := constant for w-system
            _rho   := constant for w-system

            _init_v  := initial condition for v
            _init_w  := initial condition for w
            _init_z0 := initial condition for z0
            _init_z1 := initial condition for z1

            _g1 := feedback function for z-system
            _g2 := feedback function for w-system
            _m  := feedback function for w-system

            _use_null     := turn on and off null control
            _use_coupling := turn on and off coupling
    """

    def __init__(self, num_points, delta_x, num_steps, delta_t,
                 gamma, rho,
                 init_v, init_w, init_z0, init_z1,
                 g1, g2, m,
                 use_null, use_coupling):
        self._num_points = num_points
        self._delta_x = delta_x
        self._num_steps = num_steps
        self._delta_t = delta_t

        self._gamma = gamma
        self._rho = rho

        self._init_v = init_v
        self._init_w = init_w
        self._init_z0 = init_z0
        self._init_z1 = init_z1

        self._g1 = g1
        self._g2 = g2
        self._m = m

        self._use_null = use_null
        self._use_coupling = use_coupling

    @classmethod
    def setup(cls, num_points, delta_x, num_steps, delta_t,
              gamma, rho,
              init_v, init_w, init_z0, init_z1,
              g1, g2, m,
              use_null, use_coupling):
        """Setup the system completely"""

        new = cls(num_points, delta_x, num_steps, delta_t,
                  gamma, rho,
                  init_v, init_w, init_z0, init_z1,
                  g1, g2, m,
                  use_null, use_coupling)
        new._build()

        return new

    @property
    def vec_num_points(self):
        """Get the number of points in a vec"""

        return 2 * self._num_points

    def _build(self):
        """Build the feedback control system"""

        print("Build the solver system")
        self._build_solver()

        print("Build the FEM system")
        self._build_fem()

        print("Build the w-system")
        self._build_a00_w()
        self._build_a01_w()
        self._build_a02_w()
        self._build_a11_w()
        self._build_a22_w()
        self._build_ml_w()
        self._build_m_w()
        self._build_a_w()
        self._build_e_w()

        print("Build the z-system")
        self._build_a10_z()
        self._build_a_z()
        self._build_a00_z()
        self._build_m_z()

        print("Build the init_w")
        self._build_init_w0()
        self._build_init_w1()
        self._build_init_w()

        print("Build the init_z")
        self._build_init_z0()
        self._build_init_z1()
        self._build_init_z()
        #
        # print("Build the feedback_w")
        # self._build_n_square_w()
        # self._build_n_cube_w()
        # self._build_f_plus_w()
        # self._build_n_w()
        #
        # print("Build the feedback_z")
        # self._build_n_z()
        # self._build_f_plus_z()
        #
        # print("Build the coupling_z")
        # self._build_a110_z()
        #
        # print("Build the coupling_w")
        # self._build_a200_w()

    def _build_solver(self):
        """Build the time-solver"""

        self._solver = solver.RK4(self._delta_t)

    def _build_fem(self):
        """Build the fem system"""

        self._fem = fem.FiniteElements.setup(self._num_points, self._delta_x)

    def split_eqn(self, u):
        """Splits the equation u into it's two parts

            Args:
                u := equation to split

            Returns:
                the two parts
        """

        u0 = u[:self.vec_num_points]
        u1 = u[self.vec_num_points:]

        return u0, u1

    def split_solution(self, u):
        """Splits the solution u into the w and z systems

            Args:
                u := equation to split

            Returns:
                the two parts
        """

        z = u[:2 * self.vec_num_points]
        w = u[2 * self.vec_num_points:]

        return z, w

    @staticmethod
    def identify_matrix(matrix, index):
        """Identify row and column of matrix corresponding to index

            Args:
                matrix := matrix to identify
                index  := index of row/column to identify
        """

        matrix[:, index] = 0.0
        matrix[index, :] = 0.0
        matrix[index, index] = 1.0

    def _build_a00_w(self):
        """Build the A_00 matrix for w-system

            A_00 -> identified for B.C.s
        """

        A = self._fem.A_00.copy().tolil()

        self.identify_matrix(A, 0)
        self.identify_matrix(A, self._num_points)

        self.A_00_w = A.tocsc()

    def _build_a01_w(self):
        """Builds the A_01 matrix for the w-system

            A_01 -> identified for B.C.s
        """

        A = self._fem.A_01.copy().tolil()

        self.identify_matrix(A, 0)
        self.identify_matrix(A, self._num_points)

        self.A_01_w = A.tocsc()

    def _build_a02_w(self):
        """Builds the A_02 matrix for the w-system

            A_02 -> identified for B.C.s
        """

        A = self._fem.A_02.copy().tolil()

        self.identify_matrix(A, 0)
        self.identify_matrix(A, self._num_points)

        self.A_02_w = A.tocsc()

    def _build_a11_w(self):
        """Builds the A_11 (Laplacian) matrix for the w-system

            A_11 -> identified for B.C.s
        """

        A = self._fem.A_11.copy().tolil()

        self.identify_matrix(A, 0)
        self.identify_matrix(A, self._num_points)

        A[0, 0] = 0.0
        A[self._num_points, self._num_points] = 0.0

        self.A_11_w = A.tocsc()

    def _build_a22_w(self):
        """Builds the A_22 (Biharmonic) matrix for the w-system

            A_22 -> identified for B.C.s
        """

        A = self._fem.A_22.copy().tolil()

        self.identify_matrix(A, 0)
        self.identify_matrix(A, self._num_points)

        self.A_22_w = A.tocsc()

    def _build_ml_w(self):
        """Builds the mass/lapacian operator matrix for the w-system

            ML_w = A_00_w + (rho * A_11_w)
        """

        M = self.A_00_w.copy()
        L = self.A_11_w.copy()

        self.ML_w = (M + (self._rho * L)).tocsc()

    def _build_m_w(self):
        """Builds the mass matrix for the w-system

            M_w = [[ML_w, None],
                   [None, Id  ]]
        """

        top = self.ML_w.copy()
        bottom = sp.identity(self.vec_num_points).tocsc()

        self.M_w = sp.bmat([[top,  None],
                            [None, bottom]]).tocsc()

    def _build_a_w(self):
        """Builds the A matrix for the w-system

            A_w = [[None, -gamma^2 * A_22_w],
                   [  Id, None]]
        """

        B = self.A_22_w.copy()

        top = -self._gamma**2 * B
        bottom = sp.identity(self.vec_num_points)

        self.A_w = sp.bmat([[None,   top],
                            [bottom, None]]).tocsc()

    def _build_e_w(self):
        """Builds the energy matrix for w-system

            E_w = [[ML_w, None],
                   [None, gamma^2 * A_22_w]]
        """

        B = self.A_22_w.copy()

        top = self.ML_w.copy()
        bottom = self._gamma**2 * B

        self.E_w = sp.bmat([[top,  None],
                            [None, bottom]]).tocsc()

    def _build_a10_z(self):
        """Build the A_10 matrix for the z-system

            A_10 -> identified for B.C.s
        """

        A = self._fem.A_10.copy().tolil()

        self.identify_matrix(A, self._num_points - 1)

        self.A_10_z = A.tocsc()

    def _build_a_z(self):
        """Build the A matrix for the z-system

            A_z = [[      0, A_10_z.transpose()],
                   [-A_10_z, 0                 ]]
        """

        A = self.A_10_z.copy()

        top = A.copy().transpose()
        bottom = -A.copy()

        self.A_z = sp.bmat([[None,   top],
                            [bottom, None]]).tocsc()

    def _build_a00_z(self):
        """Build the A_00 matrix for z-system

            A_00 -> identified for B.C.s
        """

        A = self._fem.A_00.copy().tolil()

        self.A_00_z = A.tocsc()

    def _build_m_z(self):
        """Build the 'mass' matrix for z-system

            M_z = [[A_00_z,      0],
                   [     0, A_00_z]]
        """

        A = self.A_00_z.copy()

        self.M_z = sp.bmat([[A,    None],
                            [None, A]]).tocsc()

    def _build_init_z0(self):
        """Build initial vector z0"""

        if isinstance(self._init_z0, np.ndarray):
            self.init_z0 = self._init_z0
        else:
            self.init_z0 = self._fem.build_vec(self._init_z0)

    def _build_init_z1(self):
        """Build initial vector z1"""

        if isinstance(self._init_z1, np.ndarray):
            self.init_z1 = self._init_z1
        else:
            self.init_z1 = self._fem.build_vec(self._init_z1)

    def _build_init_w0(self):
        """Build initial vector v"""

        if isinstance(self._init_v, np.ndarray):
            self.init_w0 = self._init_v
        else:
            vec = self._fem.build_vec(self._init_v)

            vec[0] = 0.0
            vec[self._num_points] = 0.0

            M = self.ML_w.copy()

            self.init_w0 = lin.spsolve(M, vec)

    def _build_init_w1(self):
        """Build initial vector w"""

        if isinstance(self._init_w, np.ndarray):
            self.init_w1 = self._init_w
        else:
            vec = self._fem.build_vec(self._init_w)

            vec[0] = 0.0
            vec[self._num_points] = 0.0

            M = self.A_22_w.copy()

            self.init_w1 = lin.spsolve(M, vec)

    def _build_init_w(self):
        """Build the initial vector for w-system"""

        vec0 = self.init_w0
        vec1 = self.init_w1

        self.init_w = np.concatenate((vec0, vec1))

    def _build_init_z(self):
        """Build initial vector for z-system"""

        vec0 = self.init_z0
        vec1 = self.init_z1

        vec = np.concatenate((vec0, vec1))

        if isinstance(self._init_z0, np.ndarray):
            self.init_z = vec
        else:
            M = self.M_z.copy()
            vec = lin.spsolve(M, vec)

            self.init_z = vec
