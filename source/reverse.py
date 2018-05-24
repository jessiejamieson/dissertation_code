import numpy as np
from source import null
from source import feedback


class ReverseControl(object):
    """Class to contain the reverse control

        Attributes:
            _num_points := the number of space points (includes endpoints)
            _delta_x    := distance between space points
            _delta_t    := amount of time to take in one timestep

            _gamma := constant for w-system
            _rho   := constant for w-system
            _T     := wall time for null control

            _init_v  := initial condition for v
            _init_w  := initial condition for w
            _init_z0 := initial condition for z0
            _init_z1 := initial condition for z1

            _g1 := feedback function for z-system
            _g2 := feedback function for w-system
            _m  := feedback function for w-system

            _epsilon   := feedback target energy is less than this
            _max_steps := max_steps to attempt to achieve target

            _use_coupling := turn on and off coupling
    """

    def __init__(self, num_points, delta_x, delta_t,
                 gamma, rho, T,
                 init_v, init_w, init_z0, init_z1,
                 g1, g2, m,
                 epsilon, max_steps,
                 use_coupling):
        self._num_points = num_points
        self._delta_x = delta_x
        self._delta_t = delta_t

        self._gamma = gamma
        self._rho = rho
        self._T = T

        self._init_v = init_v
        self._init_w = init_w
        self._init_z0 = init_z0
        self._init_z1 = init_z1

        self._g1 = g1
        self._g2 = g2
        self._m = m

        self._epsilon = epsilon
        self._max_steps = max_steps

        self._use_coupling = use_coupling

        self._system_ff = None
        self._system_fn = None
        self._system_rn = None
        self._system_rf = None

    @property
    def num_steps(self):
        """Calculate the number of steps"""

        return int(self._T / self._delta_t)

    def build_feedback(self, num_steps,
                       init_v, init_w, init_z0, init_z1,
                       use_reverse, storage_w, storage_z):
        """Build a complete feedback system as needed"""

        return feedback.FeedbackControl.\
            setup(self._num_points, self._delta_x, num_steps, self._delta_t,
                  self._gamma, self._rho,
                  init_v, init_w, init_z0, init_z1,
                  self._g1, self._g2, self._m,
                  True, self._use_coupling, use_reverse,
                  self._epsilon, self._max_steps,
                  storage_w, storage_z)

    def build_null(self, init_v, init_w, init_z0, init_z1,
                   storage_w, storage_z):
        """Build a complete null system as needed"""

        return null.NullControl.\
            setup(self._num_points, self._delta_x,
                  self.num_steps, self._delta_t,
                  self._gamma, self._rho, self._T,
                  init_v, init_w, init_z0, init_z1,
                  True, self._use_coupling,
                  storage_w, storage_z)

    def run_forward_feedback(self):
        """Run the forward feedback system"""

        system = self.build_feedback(None,
                                     self._init_v, self._init_w,
                                     self._init_z0, self._init_z1,
                                     True, None, None)

        self._system_ff = system

        return system.run()

    def energy_ff(self, u):
        """Compute the energies of u"""

        time, space = u.shape

        e_z = np.zeros((time, 1))
        e_w = np.zeros((time, 1))
        e_t = np.zeros((time, 1))

        for index in range(time):
            e_z[index], e_w[index], e_t[index] =\
                self._system_ff.energy(u[index])

        return e_z, e_w, e_t

    def run_forward_null(self, init_v, init_w, init_z0, init_z1):
        """Run the forward null system"""

        system = self.build_null(init_v, init_w, init_z0, init_z1, None, None)

        self._system_fn = system

        return system.run()

    def energy_fn(self, u):
        """Compute the energies of u"""

        time, space = u.shape

        e_z = np.zeros((time, 1))
        e_w = np.zeros((time, 1))
        e_t = np.zeros((time, 1))

        for index in range(time):
            e_z[index], e_w[index], e_t[index] = \
                self._system_fn.energy(u[index])

        return e_z, e_w, e_t

    def run_reverse_null(self, init_v, init_w, init_z0, init_z1,
                         storage_w, storage_z):
        """Run the reverse null system"""

        system = self.build_null(init_v, init_w, init_z0, init_z1,
                                 storage_w, storage_z)

        self._system_rn = system

        return system.run()

    def energy_rn(self, u, target):
        """Compute the energies of u"""

        time, space = u.shape

        e_z = np.zeros((time, 1))
        e_w = np.zeros((time, 1))
        e_t = np.zeros((time, 1))

        for index in range(time):
            e_z[index], e_w[index], e_t[index] = \
                self._system_rn.energy(u[index] - target)

        return e_z, e_w, e_t

    def run_reverse_feedback(self, num_steps, init_v, init_w, init_z0, init_z1,
                             storage_w, storage_z):
        """Run the reverse feedback system"""

        system = self.build_feedback(num_steps,
                                     init_v, init_w, init_z0, init_z1,
                                     False, storage_w, storage_z)

        self._system_rf = system

        return system.run()

    def energy_rf(self, u, target):
        """Compute the energies of u"""

        time, space = u.shape

        e_z = np.zeros((time, 1))
        e_w = np.zeros((time, 1))
        e_t = np.zeros((time, 1))

        for index in range(time):
            e_z[index], e_w[index], e_t[index] = \
                self._system_rf.energy(u[index] - target)

        return e_z, e_w, e_t

    def split_solution(self, u):
        """Split u into its 4 components"""

        vec_num_points = 2 * self._num_points

        z_sys = u[:2 * vec_num_points]
        w_sys = u[2 * vec_num_points:]

        z0 = z_sys[:vec_num_points]
        z1 = z_sys[vec_num_points:]

        v = w_sys[:vec_num_points]
        w = w_sys[vec_num_points:]

        return v, w, z0, z1

    def run(self):
        """Run the full system"""

        solution_ff, e_z_ff, e_w_ff, e_t_ff, storage_z_ff, storage_w_ff =\
            self.run_forward_feedback()

        init_fn = solution_ff[-1, :]
        init_v_fn, init_w_fn, init_z0_fn, init_z1_fn =\
            self.split_solution(init_fn)

        solution_fn, e_z_fn, e_w_fn, e_t_fn, storage_z_fn, storage_w_fn = \
            self.run_forward_null(init_v_fn, init_w_fn, init_z0_fn, init_z1_fn)

        init_rn = solution_fn[-1, :]
        init_v_rn, init_w_rn, init_z0_rn, init_z1_rn = \
            self.split_solution(init_rn)

        init_v_rn_new = np.negative(init_v_rn)
        init_z1_rn_new = np.negative(init_z1_rn)

        solution_rn, e_z_rn, e_w_rn, e_t_rn, storage_z_rn, storage_w_rn = \
            self.run_reverse_null(init_v_rn_new, init_w_rn,
                                  init_z0_rn, init_z1_rn_new,
                                  storage_w_fn, storage_z_fn)

        init_rf = solution_rn[-1, :]
        init_v_rf, init_w_rf, init_z0_rf, init_z1_rf = \
            self.split_solution(init_rf)

        num_rf_steps, sys_num_points = solution_ff.shape

        solution_rf, e_z_rf, e_w_rf, e_t_rf, storage_z_rf, storage_w_rf = \
            self.run_reverse_feedback(num_rf_steps,
                                      init_v_rf, init_w_rf,
                                      init_z0_rf, init_z1_rf,
                                      storage_w_ff, storage_z_ff)

        # TODO: Process the solutions to get the correct energies (for reverse)

        target = solution_ff[0, :]
        target_v, target_w, target_z0, target_z1 = self.split_solution(target)

        target_v_new = np.negative(target_v)
        target_z1_new = np.negative(target_z1)

        target_new_z = np.concatenate((target_z0, target_z1_new))
        target_new_w = np.concatenate((target_v_new, target_w))

        target_new = np.concatenate((target_new_z, target_new_w))

        e_z_rn_new, e_w_rn_new, e_t_rn_new = self.energy_rn(solution_rn,
                                                            target_new)
        e_z_rf_new, e_w_rf_new, e_t_rf_new = self.energy_rf(solution_rf,
                                                            target_new)

        e_z_target = np.concatenate((e_z_ff, e_z_fn, e_z_rn_new, e_z_rf_new))
        e_w_target = np.concatenate((e_w_ff, e_w_fn, e_w_rn_new, e_w_rf_new))
        e_t_target = np.concatenate((e_t_ff, e_t_fn, e_t_rn_new, e_t_rf_new))

        e_z = np.concatenate((e_z_ff, e_z_fn, e_z_rn, e_z_rf))
        e_w = np.concatenate((e_w_ff, e_w_fn, e_w_rn, e_w_rf))
        e_t = np.concatenate((e_t_ff, e_t_fn, e_t_rn, e_t_rf))

        return e_z, e_w, e_t, e_z_target, e_w_target, e_t_target,\
               e_z_ff, e_w_ff, e_t_ff, e_z_fn, e_w_fn, e_t_fn,\
               e_z_rn, e_w_rn, e_t_rn, e_z_rf, e_w_rf, e_t_rf
