#!/usr/bin/env python3
import numpy as np
import os
from filterpy.kalman import ExtendedKalmanFilter
import transforms3d as tf
import matplotlib.pyplot as plt

DEBUG = False
BEACONS_NUM = 5
AGENTS_NUM = 2


class Beacon:

    def __init__(self, id: str, x0: np.ndarray) -> None:
        if x0.shape == (3,) or x0.shape == (1, 3):
            x0 = x0.reshape((3, 1))
        if x0.shape != (3, 1):
            raise Exception("Wrong state shape!")
        self.__x = x0

    def get_pos(self) -> np.array:
        return self.__x


def getH(x_op: np.ndarray, beacons):
    """    
    pr - robot position
    pbi = beacon_i position
    J = [
        d/dx h(x) = norm(pr - pb1), 
        d/dx h(x) = norm(pr - pb2),
        ... 
        d/dx h(x) = norm(pr - pbn),
    ]
    first row of J (distance function h(x) to the beacon 1):
    -(bx - x)/((bx - x)^2 + (by - y)^2 + (bz - z)^2)^(1/2)
    -(by - y)/((bx - x)^2 + (by - y)^2 + (bz - z)^2)^(1/2)
    -(bz - z)/((bx - x)^2 + (by - y)^2 + (bz - z)^2)^(1/2)
    or switch bx,x places and remove minus in front
    """
    H = np.zeros((len(beacons), len(x_op)))
    for i, b in enumerate(beacons):
        diff = x_op[:3] - b.get_pos()
        the_norm = np.linalg.norm(diff)
        if (diff == 0.0).all():
            raise Exception("Division by zero")
        H[i][:3] = (diff / the_norm).T
    return H


def hx(x, beacons):
    """
    non-linear measurement func
    """
    h = np.zeros((len(beacons), 1))
    for i, b in enumerate(beacons):
        h[i] = np.linalg.norm(x[:3] - b.get_pos())
    return h


class Agent:
    DIM_Z = BEACONS_NUM + AGENTS_NUM - 1
    DIM_X = 9
    DIM_U = 6

    def __init__(self, data) -> None:
        self.__data = data
        self.filter = ExtendedKalmanFilter(
            dim_x=self.DIM_X, dim_z=self.DIM_Z, dim_u=self.DIM_U)
        self.filter.x = np.array([data["ref_pos"][0][0], data["ref_pos"]
                                 [0][1], data["ref_pos"][0][2], 0, 0, 0, 0, 0, 0]).reshape(9, 1)
        print(f"filter init: {self.filter.x[:3]}")
        dt = 0.01
        I = np.eye(3)
        Idt = np.eye(3) * dt
        Idt2 = .5 * np.eye(3) * dt**2
        from scipy.linalg import block_diag
        F = block_diag(I, I, I)
        F[0:3, 3:6] = Idt
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Idt2
        B[3:6, 0:3] = Idt
        B[6:9, 3:6] = Idt
        self.filter.F = F
        self.filter.B = B
        self.filter.P = np.eye(9)*1e3
        self.filter.R = np.eye(self.DIM_Z)
        self.filter.Q = np.eye(9)*1e-3
        self.g = np.array([0, 0, 9.794841972265039942e+00])
        self.trajectory = []

    def get_pos(self, groud_truth=True):
        if groud_truth:
            return self.__data["ref_pos"][i].reshape(3, 1)
        return self.filter.x[:3].copy()

    def num_data(self):
        return len(self.__data["ref_pos"])

    def kalman_update(self, beacons, agents, step_index):
        # save position
        self.trajectory.append(self.filter.x[:3].copy())

        # predict
        acc = self.__data["accel-0"][step_index]
        gyro = self.__data["gyro-0"][step_index]
        domega = gyro.copy()
        R_att = tf.euler.euler2mat(
            self.filter.x[6], self.filter.x[7], self.filter.x[8], axes='sxyz')
        acc = (R_att @ acc) + self.g
        theta = self.filter.x[7][0]
        phi = self.filter.x[6][0]
        Rw = np.array([[np.cos(theta), 0, -np.cos(phi)*np.sin(theta)],
                       [0, 1, np.sin(phi)],
                       [np.sin(theta), 0, np.cos(phi)*np.cos(theta)]])
        domega = domega * np.pi / 180
        u = np.concatenate((acc, Rw @ domega)).reshape(6, 1)
        self.filter.predict(u=u)
        if DEBUG:
            print(f"filter predict: {self.filter.x}")

        # update
        NOISE_STD = 1
        gt_dists = [
            self.__data[f"uwb-static{i}"][step_index][0] +
            np.random.normal(scale=NOISE_STD) for i in range(BEACONS_NUM)]
        
        for j in range(AGENTS_NUM+1):
            # check if self.__data has uwb-static key
            if f"uwb-agent{j+1}" not in self.__data.keys():
                continue
            gt_dists.append(
                self.__data[f"uwb-agent{j+1}"][step_index] +
                np.random.normal(scale=NOISE_STD))
            print("ADD", j+1)
        
        if DEBUG:
            print(f"True dists: {gt_dists}")
        z = np.array(gt_dists).reshape(-1, 1)
        beacons.extend(agents)
        # TODO: in second iteration H has 7 rows!!!
        self.filter.update(z, getH, hx, args=(beacons), hx_args=(beacons))
        print("update")


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


agent1_data = take_in_data("data/agent1")
agent2_data = take_in_data("data/agent2")

STARTING_POS = agent1_data["ref_pos"][0]

agent1_data["ref_pos"] = agent1_data["ref_pos"] - STARTING_POS
agent2_data["ref_pos"] = agent2_data["ref_pos"] - STARTING_POS

static_beacons = []
for i in range(BEACONS_NUM):
    x0 = agent1_data[f"uwb-static{i}"][0][1:] - STARTING_POS
    static_beacons.append(Beacon(str(i), x0))
    print(f"beacon{i}: {x0}")

Agent1 = Agent(agent1_data)
Agent2 = Agent(agent2_data)
global_agents = [Agent1, Agent2]
for i in range(Agent1.num_data()-1):
    for ag in global_agents:
        agents_without_itself = [a for a in global_agents if a is not ag]
        ag.kalman_update(static_beacons, agents_without_itself, i)

# plot agents and static_beacons in map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for beacon in static_beacons:
    position = beacon.get_pos()
    ax.scatter(position[0], position[1], position[2])
ax.plot(agent1_data["ref_pos"][:, 0], agent1_data["ref_pos"]
        [:, 1], agent1_data["ref_pos"][:, 2])
ax.plot(agent2_data["ref_pos"][:, 0], agent2_data["ref_pos"]
        [:, 1], agent2_data["ref_pos"][:, 2])
for agent in global_agents:
    traj = np.array(agent.trajectory)
    traj = traj.reshape(-1, 3)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '--')
plt.legend(["beacon1", "beacon2", "beacon3",
           "beacon4", "beacon5", "agent1", "agent2", "EKF_agent1", "EKF_agent2"])
plt.show()
