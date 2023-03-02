#!/usr/bin/env python3
"""
https://github.com/Aceinna/gnss-ins-sim

The simplest demo of Sim.
Only generate reference trajectory (pos, vel, sensor output). No algorithm.
Created on 2018-01-23
@author: dongxiaoguang
"""

import os
import math
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim
import numpy as np
import matplotlib.pyplot as plt

# globals
D2R = math.pi/180
PLOT_UWB = False
NUM_UWB_BEACONS = 5

motion_def_path = os.path.abspath('.//data//motion-profiles//')
fs = 100.0          # IMU sample frequency
fs_gps = 10.0       # GPS sample frequency
fs_mag = fs         # magnetometer sample frequency, not used for now


def test_path_gen():
    '''
    test only path generation in Sim.
    '''
    # choose a built-in IMU model, typical for IMU381
    imu_err = 'low-accuracy'
    # generate GPS and magnetometer data
    imu = imu_model.IMU(accuracy=imu_err, axis=9, gps=True)
    # mag_error = {'si': np.eye(3) + np.random.randn(3, 3)*0.1,
    #              'hi': np.array([10.0, 10.0, 10.0])*1.0}
    # imu.set_mag_error(mag_error)

    # TODO: move motions profiles somewhere else
    NUM_OF_AGENTS = 3
    for i in range(NUM_OF_AGENTS):
        # start simulation
        sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                          motion_def_path+f"//motion_agent{i+1}.csv",
                          ref_frame=1,
                          imu=imu,
                          mode=None,
                          env=None,
                          algorithm=None)

        sim.run(1)
        # save simulation data to files
        sim.results(data_dir=f'data/agent{i+1}')


def uwb_gen():
    data_dir = 'data'
    agents = os.listdir(data_dir)
    # trim agent if agent is not in name
    agents = [a for a in agents if 'agent' in a]
    ref_pos_files = [f'{data_dir}/{a}/ref_pos.csv' for a in agents]
    ref_pos = {}
    for refs in ref_pos_files:
        agent_dir = refs.split('/')[-2]
        data = np.genfromtxt(refs, delimiter=',')[1:]
        ref_pos[agent_dir] = data

    # generate uwb static beacons
    STATIC_STD_FROM_START = 150
    dict_items = ref_pos.items()
    a1, ref1 = next(iter(dict_items))
    STARTING_POS = ref1[0]
    for i in range(NUM_UWB_BEACONS):
        ref_pos[f'static{i}'] = np.full(
            ref1.shape, STARTING_POS + np.random.normal(scale=STATIC_STD_FROM_START, size=3))

    for a1, ref1 in ref_pos.items():
        for a2, ref2 in ref_pos.items():
            if ref1 is ref2:
                continue
            if "static" in a1:
                continue
            dist = np.linalg.norm(ref1 - ref2, axis=1)  # ground truth
            if "static" in a2:
                np.savetxt(f'{data_dir}/{a1}/uwb-{a2}.csv',
                           np.hstack((dist.reshape(-1, 1), ref2)), delimiter=',')
            else:
                np.savetxt(f'{data_dir}/{a1}/uwb-{a2}.csv',
                           dist.reshape(-1, 1), delimiter=',')
            if PLOT_UWB:
                plt.figure()
                plt.plot(dist)


if __name__ == '__main__':
    test_path_gen()
    uwb_gen()
