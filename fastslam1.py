import numpy as np
import random
import copy
import os
import argparse
import yaml

from world import World
from robot import Robot
# from motion_model import MotionModel
from kinematic_bicycle_model import KinematicBicycleModel
from measurement_model import MeasurementModel
from utils import absolute2relative, relative2absolute, degree2radian, visualize


if __name__ == "__main__":
    # main execution block
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", type=str, default="scene-1", help="map that robot navigates in")
    parser.add_argument("-p", "--particles", type=int, default=100, help="number of particles")
    args = parser.parse_args()

    maps = ['scene-1', 'scene-2']
    assert args.map in maps, "Please specify one of the map in {}.".format(maps)
    assert 0 < args.particles < 200, "The number of particles should be larger than 0 and smaller than a reasonable value."

    # config setup
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ROBOT = config['robot']
    SCENE = config[args.map]
    NUMBER_OF_PARTICLES = args.particles

    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, "fastslam1")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, args.map)

    # world and robot initilisation
    # world
    world = World()
    world.read_map(SCENE['map'])
    world_grid = world.get_grid()

    # robot
    (x, y, theta) = SCENE['R_init']
    R = Robot(x, y, degree2radian(theta), world_grid, ROBOT, sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()

    # particles initialize
    p = [None] * NUMBER_OF_PARTICLES
    (x, y, theta) = SCENE['p_init']
    init_grid = np.ones(SCENE['grid_size']) * ROBOT['prior_prob']
    for i in range(NUMBER_OF_PARTICLES):
        p[i] = Robot(x, y, degree2radian(theta), copy.deepcopy(init_grid), ROBOT)

    """
    motion_model = motion_model(config['motion_model'])
    measurement_model = MeasurementModel(config['measurement_model'], ROBOT['radar_range'])
    # FastSLAM1.0
    for idx, (forward, turn) in enumerate(SCENE['controls']):
        # move robot
        R.move(turn=degree2radian(turn), forward=forward)
        curr_odo = R.get_state()
        R.update_trajectory()
        # sensor reading and map update
        z_star, free_grid_star, occupy_grid_star = R.sense()
        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)
        w = np.zeros(NUMBER_OF_PARTICLES)
        for i in range(NUMBER_OF_PARTICLES):
            # Simulate a robot motion for each of these particles
            prev_pose = p[i].get_state()
            x, y, theta = motion_model.sample_motion_model(prev_odo, curr_odo, prev_pose)
            p[i].set_states(x, y, theta)
            p[i].update_trajectory()
            # Calculate particle's weights depending on robot's measurement
            z, _, _ = p[i].sense()
            w[i] = measurement_model.measurement_model(z_star, z)
            # Update occupancy grid based on the true measurements
            curr_pose = p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            p[i].update_occupancy_grid(free_grid, occupy_grid)
        # normalize
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]
        # select best particle
        estimated_R = copy.deepcopy(p[best_id])
        # Resample the particles with a sample probability proportional to the importance weight
        # Use low variance sampling method
        new_p = [None] * NUMBER_OF_PARTICLES
        J_inv = 1 / NUMBER_OF_PARTICLES
        r = random.random() * J_inv
        c = w[0]
        i = 0
        for j in range(NUMBER_OF_PARTICLES):
            U = r + j * J_inv
            while (U > c):
                i += 1
                c += w[i]
            new_p[j] = copy.deepcopy(p[i])
        p = new_p
        prev_odo = curr_odo
        visualize(R, p, estimated_R, free_grid_star, idx, "FastSLAM 1.0", output_path, False)
        """

    # TO DO:
    # FastSLAM1.0 - Kinematic Bicycle Model
    motion_model = KinematicBicycleModel(config['motion_model'])
    measurement_model = MeasurementModel(config['measurement_model'], ROBOT['radar_range'])
    for idx in range(len(SCENE['controls'])):
        # Use wall-following for control
        R.wall_follow(desired_distance=20.0, min_speed=5.0, max_speed=10.0, kp=1)
        curr_odo = R.get_state()
        R.update_trajectory()
        # Get sensor data
        z_star, free_grid_star, occupy_grid_star = R.sense()
        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)
        # Particle weight updates using the kinematic bicycle model
        w = np.zeros(NUMBER_OF_PARTICLES)
        for i in range(NUMBER_OF_PARTICLES):
            # Simulate motion for each particle
            prev_pose = p[i].get_state()
            x, y, theta = motion_model.sample_motion_model(prev_odo, curr_odo, prev_pose)
            p[i].set_states(x, y, theta)
            p[i].update_trajectory()
            # Calculate particle weights based on measurements
            z, _, _ = p[i].sense()
            w[i] = measurement_model.measurement_model(z_star, z)
            # Update occupancy grid based on measurements
            curr_pose = p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            p[i].update_occupancy_grid(free_grid, occupy_grid)
        # Normalize weights
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]
        # Select the best particle
        estimated_R = copy.deepcopy(p[best_id])
        # Resample particles using low variance sampling
        new_p = [None] * NUMBER_OF_PARTICLES
        J_inv = 1 / NUMBER_OF_PARTICLES
        r = random.random() * J_inv
        c = w[0]
        i = 0
        for j in range(NUMBER_OF_PARTICLES):
            U = r + j * J_inv
            while U > c:
                i += 1
                c += w[i]
            new_p[j] = copy.deepcopy(p[i])
        p = new_p
        prev_odo = curr_odo
        # Visualization (optional)
        visualize(R, p, estimated_R, free_grid_star, idx, "FastSLAM 1.0", output_path, True)
