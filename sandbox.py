import numpy as np
import os
from filterpy.kalman import KalmanFilter
import transforms3d as tf
import matplotlib.pyplot as plt


class Beacon:

    def __init__(self, id: str, x0: np.ndarray) -> None:
        if x0.shape == (3,) or x0.shape == (1, 3):
            x0 = x0.reshape((3, 1))
        if x0.shape != (3, 1):
            raise Exception("Wrong state shape!")
        self.__x = x0

    def get_pos(self, i=None) -> np.array:
        return self.__x


class Agent:

    def __init__(self, data) -> None:
        self.__data = data
        self.filter = KalmanFilter(dim_x=9, dim_z=5)
        self.filter.x = np.array([data["ref_pos"][0][0], data["ref_pos"]
                                 [0][1], data["ref_pos"][0][2], 0, 0, 0, 0, 0, 0]).reshape(9, 1)
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
        # self.filter.H = C # TODO: this must change each iteration
        self.filter.P = np.eye(9)*1e+3
        self.filter.R = np.eye(5)
        self.filter.Q = np.eye(9)
        self.index = 0  # current iteration of data
        self.g = np.array([0, 0, 9.794841972265039942e+00])
        self.trajectory = []

    def get_pos(self, i):
        return self.__data["ref_pos"][i].reshape(3, 1)
        return self.filter.x[:3].copy()

    def get_beacon_dists(self, beacons, step_index) -> np.array:
        ''' Simulate distance measurement
            TODO: it might be that noise i added already : ) '''
        current_ref_pos = self.__data["ref_pos"][step_index].reshape(3, 1)
        return np.array([np.linalg.norm(current_ref_pos - b.get_pos(step_index))
                         + np.random.normal() for b in beacons])

    def get_starting_pos(self):
        return self.__data["ref_pos"][0].reshape(3, 1)

    def getH(self, x_op: np.ndarray, beacons, step_index):
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
            diff = x_op[:3] - b.get_pos(step_index)
            print(diff)
            exit(0)
            if (diff == 0.0).all():
                raise Exception("Division by zero")
            H[i][:3] = (diff / np.linalg.norm(diff)).T
        return H

    def kalman_update(self, beacons, agents, step_index):
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
        print(self.filter.x[:3] - self.get_starting_pos())

        # update
        dists_beacons = self.get_beacon_dists(beacons, step_index)
        dists_agents = self.get_beacon_dists(agents, step_index)
        z = np.concatenate((dists_beacons, dists_agents))
        print(z)
        # merge list of beacons and agents
        merged_list = []
        merged_list.extend(beacons)
        merged_list.extend(agents)
        # calculate H
        H = self.getH(self.filter.x, merged_list, step_index)
        self.filter.H = H.copy()
        self.filter.update(z)
        print(self.filter.x[:3])

        # save position
        # self.trajectory.append(self.filter.x[:3].copy())
        pass


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

STARTING_POS = agent1_data["ref_pos"][0].reshape(3, 1)
BEACONS_NUM = 4

beacons = []
for i in range(BEACONS_NUM):
    x0 = agent1_data[f"uwb-static{i}"][0][1:]
    beacons.append(Beacon(str(i), x0))
    # print(f"beacon{i}: {x0}")

# each agent would have kalman filter then
agents = [Agent(agent1_data), Agent(agent2_data),]

step_i = 0
for i in range(2):
    for agent in agents:
        # agent without itself
        agents_without_itself = [a for a in agents if a is not agent]
        agent.kalman_update(beacons, agents_without_itself, step_i)
        # print(agent.get_pos() - agent.get_starting_pos())
    step_i += 1

exit()
# plot agents and beacons in map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for beacon in beacons:
    position = beacon.get_pos() - STARTING_POS
    ax.scatter(position[0], position[1], position[2])
for agent in agents:
    traj = np.array(agent.trajectory)
    traj = traj.reshape(-1, 3) - agent.get_starting_pos()
    print(traj)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])

plt.legend(["beacon1", "beacon2", "beacon3", "beacon4", "agent1", "agent2"])
plt.show()
