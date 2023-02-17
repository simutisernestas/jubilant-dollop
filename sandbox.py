import numpy as np
import os
from filterpy.kalman import KalmanFilter


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


class Beacon:

    def __init__(self, id: str, x0: np.ndarray) -> None:
        if x0.shape == (3,) or x0.shape == (1, 3):
            x0 = x0.reshape((3, 1))
        if x0.shape != (3, 1):
            raise Exception("Wrong state shape!")
        self.__x = x0
        self.__id = id
        self.__range = None

    def get_pos(self) -> np.array:
        return self.__x

    def update_range(self, distance, stamp):
        self.__range = distance

    def get_range(self, stamp):
        return self.__range

    def is_active(self):
        return self.get_range() is not None

    def get_id(self):
        return self.__id

    def discard_meas(self):
        self.__range = None


class Agent:

    def __init__(self, data) -> None:
        self.__x = x0
        self._data = data
        self.filter = KalmanFilter(dim_x=9, dim_z=2)
        print(data["ref_pos"])
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
        C = np.zeros((2, 9))
        C[0, 6] = 1
        C[1, 7] = 1
        self.filter.F = F
        self.filter.B = B
        self.filter.H = C
        self.filter.P = np.eye(9)*1e+3
        self.filter.R = np.eye(2)*1e+2
        self.filter.Q = np.eye(9)*1e-6
        self.index = 0  # current iteration of data

    def update(self, state) -> None:
        if state.shape != (6, 1):
            raise Exception("Wrong state shape!")
        self.__x = state

    def get_state(self) -> np.array:
        return self.__x

    def get_beacon_dists(self, beacons) -> np.array:
        return np.array(
            [np.linalg.norm(self.get_state()[:3] - b.get_pos()) + np.random.random() for b in beacons])

    def kalman_update(self, beacons, agents):
        return


agent1_data = take_in_data("data/agent1")
agent2_data = take_in_data("data/agent2")

STARTING_POS = agent1_data["ref_pos"][0]
BEACONS_NUM = 4

beacons = []
for i in range(BEACONS_NUM):
    x0 = agent1_data[f"uwb-static{i}"][0][1:]
    beacons.append(Beacon(str(i), x0))

# each agent would have kalman filter then
agents = [Agent(agent1_data), Agent(agent2_data),]

# for each agent
for agent in agents:
    for agent2 in agents:
        if agent is agent2:
            print("same agent")
        else:
            print("different agent")
    break

    # get beacons distances
    beacons_dists = agent.get_beacon_dists(beacons)
    # update beacons ranges
    for i, beacon in enumerate(beacons):
        beacon.update_range(beacons_dists[i], 0)
    # update agent state
    agent.kalman_update(beacons, agents)
