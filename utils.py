import matplotlib.pyplot as plt
import numpy as np
import time
import os


def make_plots(static_beacons, global_agents, save=True):
    BEACONS_NUM = len(static_beacons)
    AGENTS_NUM = len(global_agents)
    """Plot the trajectories of the agents plus static beacons."""
    print("Plotting...")
    # plot agents and static_beacons in map
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for beacon in static_beacons:
        position = beacon.get_pos()
        # increase scatter size
        ax.scatter(position[0], position[1], position[2], s=100)
    for agent in global_agents:
        ref_pos = agent.get_ref_pos()
        ax.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2])
    for agent in global_agents:
        traj = np.array(agent.trajectory)
        traj = traj.reshape(-1, 3)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '--')
    legends = [f"beacon{i}" for i in range(BEACONS_NUM)]
    legends.extend(f"agent{i}" for i in range(AGENTS_NUM))
    legends.extend(f"EKF_agent{i}" for i in range(AGENTS_NUM))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(legends)

    # create subplot for number of agents
    fig2, ax2 = plt.subplots(AGENTS_NUM, 1, figsize=(10, 10))
    for index, agent in enumerate(global_agents):
        ref_att = agent.get_ref_att()
        att = np.array(agent.attitude)
        att = att.reshape(-1, 3)
        error = np.deg2rad(ref_att[:-1]) - att
        ax2[index].plot(error)
        legends = [f"yaw", f"pitch", f"roll"]
        ax2[index].legend(legends)

    if save:
        fig.savefig(
            f"report/figures/trajectory_{int(time.time())}.png", dpi=300)
        fig2.savefig(
            f"report/figures/attitude_{int(time.time())}.png", dpi=300)


def plot_trajectories(static_beacons, global_agents):
    make_plots(static_beacons, global_agents, save=False)
    plt.show()


def take_in_data(agent_dir):
    files = os.listdir(agent_dir)
    data = {}
    for file in files:
        if not file.endswith(".csv"):
            continue
        path = os.path.join(agent_dir, file)
        without_ext = os.path.splitext(file)[0]
        data[without_ext] = np.loadtxt(path, delimiter=',', skiprows=1)
    return data


def print_error_metrics(agent):
    # compute position absolute error
    ref_pos = agent.get_ref_pos()
    traj = np.array(agent.trajectory)
    traj = traj.reshape(-1, 3)
    error = np.linalg.norm(ref_pos[:-1] - traj, axis=1)
    # compute attitude error
    ref_att = np.deg2rad(agent.get_ref_att())
    att = np.array(agent.attitude)
    att = att.reshape(-1, 3)
    att_error = np.linalg.norm(ref_att[:-1] - att, axis=1)
    print(
        f"Error agent[{agent.id}] - Attitude: {att_error.mean():.2f} Pos: {error.mean():.2f}")
