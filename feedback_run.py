import numpy as np
import matplotlib.pyplot as plt
from source import initial
from source import feedback


if __name__ == '__main__':
    print("Running the feedback system")

    use_feedback = True
    use_coupling = True
    use_reverse = True

    epsilon = 0.01
    max_steps = 100000

    num_points = 11
    delta_x = 1.0 / (num_points - 1)

    delta_t = 1.0 / 10.0 * delta_x

    num_steps = 1000

    gamma = 0.1
    rho = gamma**2

    init_v = initial.InitV()
    init_w = initial.InitW()
    init_z0 = initial.InitZ0()
    init_z1 = initial.InitZ1()

    g1 = initial.G1()
    g2 = initial.G2()
    m = initial.M()

    init_storage_w = None
    init_storage_z = None

    system = feedback.FeedbackControl.setup(num_points, delta_x,
                                            num_steps, delta_t,
                                            gamma, rho,
                                            init_v, init_w, init_z0, init_z1,
                                            g1, g2, m,
                                            use_feedback, use_coupling,
                                            use_reverse,
                                            epsilon, max_steps,
                                            init_storage_w, init_storage_z)

    solution, e_z, e_w, e_t, storage_z, storage_w = system.run()

    min_z = np.amin(e_z)
    max_z = np.amax(e_z)
    min_w = np.amin(e_w)
    max_w = np.amax(e_w)
    min_t = np.amin(e_t)
    max_t = np.amax(e_t)

    print('Min energy_z: ' + str(min_z))
    print('Max energy_z: ' + str(max_z))
    print('Min energy_w: ' + str(min_w))
    print('Max energy_w: ' + str(max_w))
    print('Min energy_t: ' + str(min_t))
    print('Max energy_t: ' + str(max_t))

    lower_plot = 0.001
    upper_plot = 0.001

    fig = plt.figure()
    full_title = 'Energy for full system'
    if use_feedback:
        full_title += ', with feedback'
    if use_coupling:
        full_title += ', with coupling'
    plt.suptitle(full_title)

    plt.subplot(1, 3, 1)
    plt.ylim([min_z - lower_plot, max_z + upper_plot])
    plt.plot(e_z)
    z_title = 'Energy for z-system'
    if use_feedback:
        z_title += ', with feedback'
    if use_coupling:
        z_title += ', with coupling'
    plt.title(z_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 2)
    plt.ylim([min_w - lower_plot, max_w + upper_plot])
    plt.plot(e_w)
    w_title = 'Energy for w-system'
    if use_feedback:
        w_title += ', with feedback'
    if use_coupling:
        w_title += ', with coupling'
    plt.title(w_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 3)
    plt.ylim([min_t - lower_plot, max_t + upper_plot])
    plt.plot(e_t)
    t_title = 'Energy for t-system'
    if use_feedback:
        t_title += ', with feedback'
    if use_coupling:
        t_title += ', with coupling'
    plt.title(t_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.show(block=False)
