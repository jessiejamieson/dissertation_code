import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin

from source import finite_elements as fem
from source import solver


class OnesFunction(object):
    """Class to contain the ones-function system"""

    def __call__(self, x):
        """Return the value of the function"""

        return 1.0


class NullControl(object):
    """Class to contain the null control system

        Attributes:
            _num_points := the number of space points (includes endpoints)
            _delta_x    := distance between space points
            _num_steps  := number of time steps to take
            _delta_t    := amount of time to take in one timestep

            _T := null control target time

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
                 T,
                 init_v, init_w, init_z0, init_z1,
                 g1, g2, m,
                 use_null, use_coupling):
        self._num_points = num_points
        self._delta_x = delta_x
        self._num_steps = num_steps
        self._delta_t = delta_t

        self._T = T

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
              T,
              init_v, init_w, init_z0, init_z1,
              g1, g2, m,
              use_null, use_coupling):
        """Setup the system completely"""

        new = cls(num_points, delta_x, num_steps, delta_t,
                  T,
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

        print("Build the null_w")
        self._build_m_w_inv()
        self._build_f_plus_w()
        self._build_n_w()
        self._build_b_w()
        self._build_abf_w()
        self._build_y0_w()

        print("Build the null_z")
        self._build_m_z_inv()
        self._build_n_z()
        self._build_f_plus_z()
        self._build_b_z()
        self._build_abf_z()
        self._build_y0_z()

        print("Build the coupling_z")
        self._build_a110_z()

        print("Build the coupling_w")
        self._build_a200_w()

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

    def _build_m_w_inv(self):
        """Build the inverse of M_w for w-system control"""

        M = self.M_w.copy()

        self.M_w_inv = lin.inv(M).tocsc()

    def _build_f_plus_w(self):
        """Build the F_plus for the w-system"""

        F = np.zeros((2, self.vec_num_points))

        F[0, self.vec_num_points - 1] = 1.0
        F[1, self._num_points - 1] = 1.0

        self.F_plus_w = F

    def _build_n_w(self):
        """Build the N_w matrix"""

        F = self.F_plus_w.copy()
        B = self.A_22_w.copy()

        A = -self._gamma**2 * B
        A_inv = lin.inv(A)

        Nt = F * A_inv

        N = Nt.transpose()
        fill = np.zeros(N.shape)

        self.N_w = np.concatenate((N, fill), axis=0)

    def _build_b_w(self):
        """Build the B matrix for null control in w-system"""

        N = self.N_w.copy()
        B = self.A_22_w.copy()
        fill = np.zeros(B.shape)

        A = sp.bmat([[B,    None],
                     [None, fill]]).tocsc()

        self.B_w = A * N

    def _build_abf_w(self):
        """Build the ABF (exponent) for the exp function for
            w-system control

            ABF_w = M_w^-1 (A_w + B_w*F_plus_w)
        """

        A = self.A_w.copy()
        B = self.B_w.copy()
        F = self.F_plus_w.copy()
        M = self.M_w_inv.copy()

        BF = sp.csc_matrix(np.matmul(B, F))
        ABF = A + BF

        self.ABF_w = M * ABF

    def exp_w(self, t):
        """Computes e^(A + BF)t for w-system:

            f(t) = exp(inv(M_w)(A_w + B_w * F_plus_w)t )

            Args:
                t := time

            Returns
                f(t)
        """

        ABF = self.ABF_w
        exp = ABF * t

        return lin.expm(exp)

    def _build_y0_w(self):
        """Build the y0 term for w-system control"""

        I = sp.identity(2 * self.vec_num_points)
        A = self.A_w.copy()
        M = self.M_w_inv.copy()
        T = self._T
        MA = M * A

        exp_ABF_T = self.exp_w(T)
        exp_A_T = lin.expm(-T * MA)
        exp = exp_A_T * exp_ABF_T
        I_exp = I - exp

        y0 = self.init_w.copy()
        y0_new = np.reshape(y0, (2 * self.vec_num_points, 1))

        sol = lin.spsolve(I_exp, y0_new)

        self.y0_w = np.reshape(sol, (2 * self.vec_num_points, 1))

    def u_w(self, t):
        """Computes the u(t) term for the w-system control

            u(t) = F_plus_w * exp_w(t) * y0_w

            Args:
                t := time

            Returns:
                u(t)
        """

        F = self.F_plus_w
        y0 = self.y0_w
        exp = self.exp_w(t)

        prod = F * exp

        return np.matmul(prod, y0)

    def null_w(self, t):
        """Computes the null control term for w-system

            N(t) = B_w * u_w(t)

            Args:
                t := time

            Returns:
                N(t)
        """

        B = self.B_w
        u = self.u_w(t)

        null = np.matmul(B, u)

        return np.reshape(null, (2 * self.vec_num_points, ))

    def control_w(self, t, w):
        """Determines the control term for w

            Args:
                t := current timestep value
                w := current value of system

            Returns:
                w modified by the control term
        """

        if self._use_null:
            null = self.null_w(t)

            return w + null, null
        else:
            null = [0.0, 0.0, 0.0, 0.0]

            return w, null

    def _build_m_z_inv(self):
        """Build the inverse of M for z-system"""

        M = self.M_z.copy()

        self.M_z_inv = lin.inv(M).tocsc()

    def _build_n_z(self):
        """Builds the N_z vector

            l2 projection of ones-function
        """

        f = OnesFunction()

        self.N_z = self._fem.l2_project(f, identify=False)

    def _build_f_plus_z(self):
        """Builds the F_plus vector for z-system"""

        N = self.N_z.copy()
        A = self.A_10_z.copy().transpose().tocsc()

        prod = np.transpose(-N) * A

        left = np.zeros((1, self.vec_num_points))
        right = np.reshape(prod, (1, self.vec_num_points))

        self.F_plus_z = -np.concatenate((left, right), axis=1)

    def _build_b_z(self):
        """Builds the B matrix for z-system control

            B = -[0, A_10_z * N_z]^t
        """

        A = self.A_10_z.copy()
        N = self.N_z.copy()

        fill = np.zeros((self.vec_num_points, ))
        b = A * N

        self.B_z = np.concatenate((fill, b))

    def _build_abf_z(self):
        """Build the ABF (exponent) for the exp function for
            z-system control

            ABF_z = M_z^-1 (A_z + B_z*F_plus_z)
        """

        A = self.A_z.copy()
        B = self.B_z.copy()
        F = self.F_plus_z.copy()
        M = self.M_z_inv.copy()

        BF = sp.csc_matrix(B * F)
        ABF = A + BF

        self.ABF_z = M * ABF

    def exp_z(self, t):
        """Computes e^(A + BF)t for z-system:

            f(t) = exp(inv(M_z)(A_z + B_z * F_plus_z)t )

            Args:
                t := time

            Returns
                f(t)
        """

        ABF = self.ABF_z
        exp = ABF * t

        return lin.expm(exp)

    def _build_y0_z(self):
        """Build the y0 term for z-system control"""

        I = sp.identity(2 * self.vec_num_points)
        A = self.A_z.copy()
        M = self.M_z_inv.copy()
        T = self._T
        MA = M * A

        exp_ABF_T = self.exp_z(T)
        exp_A_T = lin.expm(-T * MA)
        exp = exp_A_T * exp_ABF_T
        I_exp = (I - exp).tocsc()

        y0 = self.init_z.copy()

        sol = lin.spsolve(I_exp, y0)

        self.y0_z = np.reshape(sol, (2 * self.vec_num_points, 1))

    def u_z(self, t):
        """Computes the u(t) term for the z-system control

            u(t) = F_plus_z * exp_z(t) * y0_z

            Args:
                t := time

            Returns:
                u(t)
        """

        F = self.F_plus_z
        y0 = self.y0_z
        exp = self.exp_z(t)

        prod = F * exp

        return np.matmul(prod, y0)[0, 0]

    def null_z(self, t):
        """Computes the null control term for z-system

            N(t) = B_z * u_z(t)

            Args:
                t := time

            Returns:
                N(t)
        """

        B = self.B_z
        u = self.u_z(t)

        null = np.matmul(B, u)

        return np.reshape(null, (2 * self.vec_num_points, ))

    def control_z(self, t, z):
        """Determines the control term for z

            Args:
                t := current timestep value
                z := current value of system

            Returns:
                z modified by the control term
        """

        if self._use_null:
            null = self.null_z(t)

            return z + null, null
        else:
            null = [0.0, 0.0, 0.0, 0.0]

            return z, null

    def _build_a200_w(self):
        """Build the A_200 tensor for w-system"""

        self.A_200_w = self._fem.A_200.copy()

    def _coupling_w(self, z, w):
        """Returns the coupling term for w-system

            Args:
                z := current z-system vector
                w := current w-system vector

            Returns:
                coupling term
        """

        array = self.A_200_w

        z0, z1 = self.split_eqn(z)
        w0, w1 = self.split_eqn(w)

        coupling = np.zeros((self.vec_num_points, ))
        zeros = np.zeros((self.vec_num_points, ))

        for index, A_k in enumerate(array):
            temp = A_k * z0
            term = np.dot(w1, temp)
            coupling[index] = -term

        return np.concatenate((coupling, zeros))

    def coupling_w(self, t, v, z, w):
        """Add coupling term for w-system

            Args:
                t := current time
                v := current vector
                z := current z-system vec
                w := current w-system vec

            Returns:
                w modified by coupling
        """

        if self._use_coupling:
            coupling = self._coupling_w(z, w)
            return v + coupling
        else:
            return v

    def _build_a110_z(self):
        """Build the A_110 tensor for z-system"""

        self.A_110_z = self._fem.A_110.copy()

    def _coupling_z(self, u):
        """Returns the coupling term for z-system

            Args:
                u := the w-system vector

            Returns:
                coupling term
        """

        array = self.A_110_z
        v, w = self.split_eqn(u)

        coupling = np.zeros((self.vec_num_points, ))
        zeros = np.zeros((self.vec_num_points, ))

        for index, A_k in enumerate(array):
            temp = A_k * w
            coupling[index] = np.dot(v, temp)

        return np.concatenate((coupling, zeros))

    def coupling_z(self, t, z, w):
        """Add coupling term for z-system

            Args:
                t := current time
                z := current z-system vec
                w := current w-system vec

            Returns:
                z modified by coupling
        """

        if self._use_coupling:
            coupling = self._coupling_z(w)
            return z + coupling
        else:
            return z
