import numpy as np
import matplotlib.pyplot as plt
from source import initial
from source import reverse


if __name__ == '__main__':
    print("Running the reverse system")

    use_coupling = False

    epsilon = 0.0001
    max_steps = 100000

    num_points = 11
    delta_x = 1.0 / (num_points - 1)

    delta_t = 1.0 / 10.0 * delta_x

    gamma = 0.1
    rho = gamma**2
    T = 20.0

    init_v = initial.InitV()
    init_w = initial.InitW()
    init_z0 = initial.InitZ0()
    init_z1 = initial.InitZ1()

    g1 = initial.G1()
    g2 = initial.G2()
    m = initial.M()

    system = reverse.ReverseControl(num_points, delta_x, delta_t,
                                    gamma, rho, T,
                                    init_v, init_w, init_z0, init_z1,
                                    g1, g2, m,
                                    epsilon, max_steps,
                                    use_coupling)

    e_z, e_w, e_t, e_z_target, e_w_target, e_t_target, e_z_ff, e_w_ff, e_t_ff, e_z_fn, e_w_fn, e_t_fn, e_z_rn, e_w_rn, e_t_rn, e_z_rf, e_w_rf, e_t_rf = system.run()

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
    full_title = 'Energy for full system reverse control'
    if use_coupling:
        full_title += ', with coupling'
    plt.suptitle(full_title)

    plt.subplot(1, 3, 1)
    # plt.ylim([min_z - lower_plot, max_z + upper_plot])
    plt.plot(e_z)
    z_title = 'Energy for z-system reverse control'
    if use_coupling:
        z_title += ', with coupling'
    plt.title(z_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 2)
    # plt.ylim([min_w - lower_plot, max_w + upper_plot])
    plt.plot(e_w)
    w_title = 'Energy for w-system reverse control'
    if use_coupling:
        w_title += ', with coupling'
    plt.title(w_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 3)
    # plt.ylim([min_t - lower_plot, max_t + upper_plot])
    plt.plot(e_t)
    t_title = 'Energy for t-system reverse control'
    if use_coupling:
        t_title += ', with coupling'
    plt.title(t_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.show(block=False)

    fig = plt.figure()
    full_title = 'Energy for full system reverse control, target'
    if use_coupling:
        full_title += ', with coupling'
    plt.suptitle(full_title)

    plt.subplot(1, 3, 1)
    # plt.ylim([min_z - lower_plot, max_z + upper_plot])
    plt.plot(e_z_target)
    z_title = 'Energy for z-system reverse control, target'
    if use_coupling:
        z_title += ', with coupling'
    plt.title(z_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 2)
    # plt.ylim([min_w - lower_plot, max_w + upper_plot])
    plt.plot(e_w_target)
    w_title = 'Energy for w-system reverse control, target'
    if use_coupling:
        w_title += ', with coupling'
    plt.title(w_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 3)
    # plt.ylim([min_t - lower_plot, max_t + upper_plot])
    plt.plot(e_t_target)
    t_title = 'Energy for t-system reverse control, target'
    if use_coupling:
        t_title += ', with coupling'
    plt.title(t_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.show(block=False)

    fig = plt.figure()
    full_title = 'Energy for full system reverse control, forward feedback'
    if use_coupling:
        full_title += ', with coupling'
    plt.suptitle(full_title)

    plt.subplot(1, 3, 1)
    # plt.ylim([min_z - lower_plot, max_z + upper_plot])
    plt.plot(e_z_ff)
    z_title = 'Energy for z-system reverse control, forward feedback'
    if use_coupling:
        z_title += ', with coupling'
    plt.title(z_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 2)
    # plt.ylim([min_w - lower_plot, max_w + upper_plot])
    plt.plot(e_w_ff)
    w_title = 'Energy for w-system reverse control, forward feedback'
    if use_coupling:
        w_title += ', with coupling'
    plt.title(w_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 3)
    # plt.ylim([min_t - lower_plot, max_t + upper_plot])
    plt.plot(e_t_ff)
    t_title = 'Energy for t-system reverse control, forward feedback'
    if use_coupling:
        t_title += ', with coupling'
    plt.title(t_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.show(block=False)

    fig = plt.figure()
    full_title = 'Energy for full system reverse control, forward null'
    if use_coupling:
        full_title += ', with coupling'
    plt.suptitle(full_title)

    plt.subplot(1, 3, 1)
    # plt.ylim([min_z - lower_plot, max_z + upper_plot])
    plt.plot(e_z_fn)
    z_title = 'Energy for z-system reverse control, forward null'
    if use_coupling:
        z_title += ', with coupling'
    plt.title(z_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 2)
    # plt.ylim([min_w - lower_plot, max_w + upper_plot])
    plt.plot(e_w_fn)
    w_title = 'Energy for w-system reverse control, forward null'
    if use_coupling:
        w_title += ', with coupling'
    plt.title(w_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 3)
    # plt.ylim([min_t - lower_plot, max_t + upper_plot])
    plt.plot(e_t_fn)
    t_title = 'Energy for t-system reverse control, forward null'
    if use_coupling:
        t_title += ', with coupling'
    plt.title(t_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.show(block=False)

    fig = plt.figure()
    full_title = 'Energy for full system reverse control, reverse null'
    if use_coupling:
        full_title += ', with coupling'
    plt.suptitle(full_title)

    plt.subplot(1, 3, 1)
    # plt.ylim([min_z - lower_plot, max_z + upper_plot])
    plt.plot(e_z_rn)
    z_title = 'Energy for z-system reverse control, reverse null'
    if use_coupling:
        z_title += ', with coupling'
    plt.title(z_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 2)
    # plt.ylim([min_w - lower_plot, max_w + upper_plot])
    plt.plot(e_w_rn)
    w_title = 'Energy for w-system reverse control, reverse null'
    if use_coupling:
        w_title += ', with coupling'
    plt.title(w_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 3)
    # plt.ylim([min_t - lower_plot, max_t + upper_plot])
    plt.plot(e_t_rn)
    t_title = 'Energy for t-system reverse control, reverse null'
    if use_coupling:
        t_title += ', with coupling'
    plt.title(t_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.show(block=False)

    fig = plt.figure()
    full_title = 'Energy for full system reverse control, reverse feedback'
    if use_coupling:
        full_title += ', with coupling'
    plt.suptitle(full_title)

    plt.subplot(1, 3, 1)
    # plt.ylim([min_z - lower_plot, max_z + upper_plot])
    plt.plot(e_z_rf)
    z_title = 'Energy for z-system reverse control, reverse feedback'
    if use_coupling:
        z_title += ', with coupling'
    plt.title(z_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 2)
    # plt.ylim([min_w - lower_plot, max_w + upper_plot])
    plt.plot(e_w_rf)
    w_title = 'Energy for w-system reverse control, reverse feedback'
    if use_coupling:
        w_title += ', with coupling'
    plt.title(w_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.subplot(1, 3, 3)
    # plt.ylim([min_t - lower_plot, max_t + upper_plot])
    plt.plot(e_t_rf)
    t_title = 'Energy for t-system reverse control, reverse feedback'
    if use_coupling:
        t_title += ', with coupling'
    plt.title(t_title)
    plt.xlabel('timestep')
    plt.ylabel('energy')

    plt.show(block=False)
