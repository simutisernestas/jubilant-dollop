import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import math
import os


@njit
def euler2mat(ai, aj, ak):
    i = 0
    j = 1
    k = 2

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    M[i, i] = cj*ck
    M[i, j] = sj*sc-cs
    M[i, k] = sj*cc+ss
    M[j, i] = cj*sk
    M[j, j] = sj*ss+cc
    M[j, k] = sj*cs-sc
    M[k, i] = -sj
    M[k, j] = cj*si
    M[k, k] = cj*ci
    return M


def plot_trajectories(static_beacons, global_agents):
    BEACONS_NUM = len(static_beacons)
    AGENTS_NUM = len(global_agents)
    """Plot the trajectories of the agents plus static beacons."""
    print("Plotting...")
    # plot agents and static_beacons in map
    fig = plt.figure()
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

    # TODO: test
    # plot attitude
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for agent in global_agents:
        agent = global_agents[1]
        ref_att = agent.get_ref_att()
        att = np.array(agent.attitude)
        att = att.reshape(-1, 3)
        error = np.deg2rad(ref_att[:-1]) - att
        ax2.plot(error)
        legends = [f"roll", f"pitch", f"yaw"]
        ax2.legend(legends)
        break

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
