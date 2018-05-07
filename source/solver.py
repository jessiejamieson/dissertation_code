class RK4(object):
    """Class to contain a Runge-Kutta 4th Order method:

        Solves d/dt u = f(u)

        Attributes:
            _delta_t := timestep length
    """

    def __init__(self, delta_t):
        self._delta_t = delta_t

        self.storagez = []
        self.storagew = []

    def make_storage(self):
        """Create/reset a storage"""

        self.storagez = []
        self.storagew = []

    def k1(self, f, t, u, usereverse=False):
        """First stage of time stepper

            Args:
                f := function to step
                t := time
                u := previous value

                usereverse := decide if reverse storage is used

            Returns:
                K1 stage value
        """

        if usereverse:
            output, storagez, storagew = f(t, u)

            self.storagez.append(storagez)
            self.storagew.append(storagew)

            return output
        else:
            return f(t, u)

    def k2(self, f, t, u, k1, usereverse=False):
        """Second stage of time stepper

            Args:
                f  := function to step
                t  := time
                u  := previous value
                k1 := k1 stage value

                usereverse := decide if reverse storage is used

            Returns:
                K2 stage
        """

        h = self._delta_t / 2.0

        v = u + h * k1
        tt = t + h

        if usereverse:
            output, storagez, storagew = f(tt, v)

            self.storagez.append(storagez)
            self.storagew.append(storagew)

            return output
        else:
            return f(tt, v)

    def k3(self, f, t, u, k2, usereverse=False):
        """Third stage of time stepper

            Args:
                f  := function to step
                t  := time
                u  := previous value
                k2 := k2 stage value

                usereverse := decide if reverse storage is used

            Returns:
                K3 stage
        """

        h = self._delta_t / 2.0

        v = u + h * k2
        tt = t + h

        if usereverse:
            output, storagez, storagew = f(tt, v)

            self.storagez.append(storagez)
            self.storagew.append(storagew)

            return output
        else:
            return f(tt, v)

    def k4(self, f, t, u, k3, usereverse=False):
        """Fourth stage of time stepper

            Args:
                f  := function to step
                t  := time
                u  := previous value
                k3 := k3 stage value

                usereverse := decide if reverse storage is used

            Returns:
                K4 stage
        """

        v = u + self._delta_t * k3
        tt = t + self._delta_t

        if usereverse:
            output, storagez, storagew = f(tt, v)

            self.storagez.append(storagez)
            self.storagew.append(storagew)

            return output
        else:
            return f(tt, v)

    @staticmethod
    def combine(k1, k2, k3, k4):
        """Combine the k1, k2, k3, k4

            Args:
                k1 := k1 stage value
                k2 := k2 stage value
                k3 := k3 stage value
                k4 := k4 stage value

            Returns:
                K combination
        """

        return (k1 + (k2 * 2.0) + (k3 * 2.0) + k4)

    def new(self, u, k1, k2, k3, k4):
        """Create new step from the 4 stages

            Args:
                u  := current step value
                k1 := k1 stage value
                k2 := k2 stage value
                k3 := k3 stage value
                k4 := k4 stage value

            Returns:
                New step value
        """

        v = self.combine(k1, k2, k3, k4)

        return u + (self._delta_t / 6.0) * v

    def __call__(self, f, t, u, usereverse=False):
        """Make stepping forward one step the call

            Args:
                f  := function to step
                t  := time
                u  := previous value

                usereverse := decide if reverse storage is used

            Returns:
                new_step if usereverse is false
                new_step, z_values, w_values if usereverse is true
        """

        if usereverse:
            self.make_storage()

        k1 = self.k1(f, t, u, usereverse)
        k2 = self.k2(f, t, u, k1, usereverse)
        k3 = self.k3(f, t, u, k2, usereverse)
        k4 = self.k4(f, t, u, k3, usereverse)

        if usereverse:
            return self.new(u, k1, k2, k3, k4), self.storagez, self.storagew
        else:
            return self.new(u, k1, k2, k3, k4)
