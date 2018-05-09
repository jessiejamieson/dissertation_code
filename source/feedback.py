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


class SquareFunction(object):
    """Class to contain the square-function system"""

    def __init__(self, diff):
        self._diff = diff

    @staticmethod
    def diff_0(x):
        """Square function with 0 derivatives"""

        return x**2 / 2.0

    @staticmethod
    def diff_1(x):
        """Square function with 1 derivatives"""

        return x

    @staticmethod
    def diff_2(x):
        """Square function with 2 derivatives"""

        return 1.0

    def __call__(self, x):
        """Call the correct function"""

        if self._diff == 0:
            return self.diff_0(x)
        elif self._diff == 1:
            return self.diff_1(x)
        elif self._diff == 2:
            return self.diff_2(x)


class CubeFunction(object):
    """Class to contain the cube-function system"""

    def __init__(self, diff):
        self._diff = diff

    @staticmethod
    def diff_0(x):
        """Cube function with 0 derivatives"""

        return x**3 / 6.0 - x**2 / 2.0

    @staticmethod
    def diff_1(x):
        """Cube function with 1 derivatives"""

        return x**2 / 2.0 - x

    @staticmethod
    def diff_2(x):
        """Cube function with 2 derivatives"""

        return x - 1.0

    def __call__(self, x):
        """Call the correct function"""

        if self._diff == 0:
            return self.diff_0(x)
        elif self._diff == 1:
            return self.diff_1(x)
        elif self._diff == 2:
            return self.diff_2(x)


class FeedbackControl(object):
    """Class to contain the feedback control system

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

            _use_feedback := turn on and off feedback
            _use_coupling := turn on and off coupling
    """

    def __init__(self, num_points, delta_x, num_steps, delta_t,
                 gamma, rho,
                 init_v, init_w, init_z0, init_z1,
                 g1, g2, m,
                 use_feedback, use_coupling):
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

        self._use_feedback = use_feedback
        self._use_coupling = use_coupling

    @classmethod
    def setup(cls, num_points, delta_x, num_steps, delta_t,
              gamma, rho,
              init_v, init_w, init_z0, init_z1,
              g1, g2, m,
              use_feedback, use_coupling):
        """Setup the system completely"""

        new = cls(num_points, delta_x, num_steps, delta_t,
                  gamma, rho,
                  init_v, init_w, init_z0, init_z1,
                  g1, g2, m,
                  use_feedback, use_coupling)
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

        print("Build the feedback_w")
        self._build_n_square_w()
        self._build_n_cube_w()
        self._build_f_plus_w()
        self._build_n_w()

        print("Build the feedback_z")
        self._build_n_z()
        self._build_f_plus_z()

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

        self.identify_matrix(A, 0)
        self.identify_matrix(A, self._num_points)

        self.A_10_z = A.tocsc()

    def _build_a_z(self):
        """Build the A matrix for the z-system

            A_z = [[      0, A_10_z.transpose()],
                   [-A_10_z, 0                 ]]
        """

        A = self.A_10_z.copy()

        A_top = A.copy().transpose()
        A_bottom = -A.copy()

        self.A_z = sp.bmat([[None,     A_top],
                            [A_bottom, None]]).tocsc()

    def _build_a00_z(self):
        """Build the A_00 matrix for z-system

            A_00 -> identified for B.C.s
        """

        A = self._fem.A_00.copy().tolil()

        self.identify_matrix(A, 0)
        self.identify_matrix(A, self._num_points)

        self.A_00_z = A.tocsc()

    def _build_m_z(self):
        """Build the 'mass' matrix for z-system

            M_z = [[A_00_z,      0],
                   [     0, A_00_z]]
        """

        A = self.A_00_z.copy()

        self.M_z = sp.bmat([[A,    None],
                            [None, A]]).tocsc()

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

            vec[0] = 0.0
            vec[self._num_points] = 0.0
            vec[2 * self._num_points] = 0.0
            vec[3 * self._num_points] = 0.0

            self.init_z = vec

    def _build_n_square_w(self):
        """Builds the N_w vector corresponding to the square function"""

        f0 = SquareFunction(0)
        f1 = SquareFunction(1)
        f2 = SquareFunction(2)

        vec0 = self._fem.build_vec(f0, diff=0)
        vec1 = self._fem.build_vec(f1, diff=1)
        vec2 = self._fem.build_vec(f2, diff=2)

        self.N_square_w = vec0 + vec1 + vec2

    def _build_n_cube_w(self):
        """Builds the N_w vector corresponding to the cube function"""

        f0 = CubeFunction(0)
        f1 = CubeFunction(1)
        f2 = CubeFunction(2)

        vec0 = self._fem.build_vec(f0, diff=0)
        vec1 = self._fem.build_vec(f1, diff=1)
        vec2 = self._fem.build_vec(f2, diff=2)

        self.N_cube_w = vec0 + vec1 + vec2

    def _build_f_plus_w(self):
        """Builds the F_plus vector for w-system"""

        vec = np.zeros((2, self.vec_num_points))

        vec[0, -1] = 1.0
        vec[1, self._num_points - 1] = 1.0

        self.F_plus_w = vec

    def _build_n_w(self):
        """Builds the N_w matrix"""

        F_plus = self.F_plus_w.copy()
        B = self.A_22_w.copy()

        Bt = B.transpose().tocsc()
        Bt_inv = lin.inv(Bt)
        Nt = F_plus * Bt_inv

        self.N_w = Nt.transpose()

    def g_w(self, w):
        """Builds the G_w function for the w-system control

            Args:
                w := current w-system vec

            Returns:
                g_w(w) result vector
        """

        F_plus = self.F_plus_w

        prod = np.matmul(F_plus, np.reshape(w, (self.vec_num_points, 1)))

        m = self._m(prod[0])
        g = self._g2(prod[1])

        return np.array([-m, -g])

    def feedback_w(self, w):
        """Feedback control term for w-system

            Args:
                w := current value of system

            Returns:
                control term for w-system
        """

        N = self.N_w

        v = w[:self.vec_num_points]

        g_w = self.g_w(v)

        left = np.zeros((self.vec_num_points, 1))
        right = np.matmul(N, g_w)

        value = np.concatenate((left, right))

        return np.reshape(value, (2 * self.vec_num_points, ))

    def control_w(self, t, w):
        """Determines the control term for w

            Args:
                t := current timestep value
                w := current value of system

            Returns:
                w modified by the control term
        """

        if self._use_feedback:
            feedback = self.feedback_w(w)

            return w - feedback, feedback
        else:
            feedback = [0.0, 0.0, 0.0, 0.0]

            return w, feedback

    def _build_n_z(self):
        """Builds the N_z vector

            l2 projection of ones-function
        """

        f = OnesFunction()

        self.N_z = self._fem.l2_project(f, identify=False)

    def _build_f_plus_z(self):
        """Builds the F_plus vector for z-system

            F_plus_z = [-N^t * A^t, 0]
        """

        A = self.A_10_z
        N = self.N_z

        left = np.zeros((self.vec_num_points, 1))
        right = np.reshape(-np.transpose(N) * A.transpose(),
                           (self.vec_num_points, 1))

        self.F_plus_z = np.concatenate((left, right))

    def feedback_z(self, z):
        """Feedback control term for z-system

            Args:
                z := current value of system

            Returns:
                feedback control term
        """

        F_plus = self.F_plus_z
        N = self.N_z

        prod = np.dot(z, F_plus)[0]

        top = np.zeros((self.vec_num_points, ))
        bottom = self._g1(prod) * N

        return np.concatenate((bottom, top))

    def control_z(self, t, z):
        """Determines the control term for z

            Args:
                t := current timestep value
                z := current value of system

            Returns:
                z modified by the control term
        """

        if self._use_feedback:
            feedback = self.feedback_z(z)

            return z - feedback, feedback
        else:
            feedback = [0.0, 0.0, 0.0, 0.0]

            return z, feedback

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

    def rhs_w(self, t, z, w):
        """Computes the right hand side of equation for w-system

            Args:
                t := current time
                z := current z-system vec
                w := current w-system vec

            Returns:
                w rhs
        """

        M = self.M_w
        A = self.A_w

        v0, storage = self.control_w(t, w)

        v1 = A * v0

        b = self.coupling_w(t, v1, z, w)

        return lin.spsolve(M, b), storage

    def rhs_z(self, t, z, w):
        """Computes the right hand side of equation for z-system

            Args:
                t := current time
                z := current z-system vec
                w := current w-system vec

            Returns:
                z rhs
        """

        M = self.M_z
        A = self.A_z

        v0, storage = self.control_z(t, z)

        v1 = A * v0

        b = self.coupling_z(t, v1, w)

        return lin.spsolve(M, b), storage

    def rhs(self, t, u):
        """Computes the right hand side of whole system

            Args:
                t := current time
                u := current system vec

            Returns:
                rhs
        """

        z, w = self.split_solution(u)

        new_z, storage_z = self.rhs_z(t, z, w)
        new_w, storage_w = self.rhs_w(t, z, w)

        return np.concatenate((new_z, new_w)), storage_z, storage_w

    def step(self, t, u):
        """Computes the new step for whole system

            Args:
                t := current time
                u := current system vec

            Returns:
                new system vec, storage_z, storage_w
        """

        return self._solver(self.rhs, t, u, usereverse=True)

    def energy_w(self, w):
        """Calculates the energy for w-system

            Args:
                w := current system vec

            Returns:
                energy for w
        """

        E = self.E_w
        v = E * w
        value = np.dot(w, v)

        return 0.5 * value

    def energy_z(self, z):
        """Calculates the energy for z-system

            Args:
                z := current system vec

            Returns:
                energy for z
        """

        M = self.M_z
        v = M * z
        value = np.dot(z, v)

        return 0.5 * value

    def energy(self, u):
        """Calculates all of the energies

            Args:
                u := current system vec

            Returns:
                energy for z, energy for w, total energy
        """

        z, w = self.split_solution(u)

        e_z = self.energy_z(z)
        e_w = self.energy_w(w)

        return e_z, e_w, e_z + e_w

    def run(self):
        """Run the solver"""

        storage_z = []
        storage_w = []

        solution = np.zeros((self._num_steps + 1, 4 * self.vec_num_points))
        solution[0, :] = np.concatenate((self.init_z, self.init_w))

        e_z = np.zeros((self._num_steps + 1, 1))
        e_w = np.zeros((self._num_steps + 1, 1))
        e_t = np.zeros((self._num_steps + 1, 1))

        e_z[0], e_w[0], e_t[0] = self.energy(solution[0, :])

        if self._use_feedback:
            step_str = "Step feedback: "
        else:
            step_str = "Step: "

        t = 0.0
        for index in range(self._num_steps):
            print(step_str + str(index + 1))

            t += self._delta_t
            u = solution[index, :]

            new, s_z, s_w = self.step(t, u)

            solution[index + 1, :] = new
            storage_z.extend(s_z)
            storage_w.extend(s_w)

            e_z[index + 1], e_w[index + 1], e_t[index + 1] = self.energy(new)

        return solution, e_z, e_w, e_t, storage_z, storage_w
