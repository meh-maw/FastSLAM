import random
import numpy as np

from utils import *


class Robot(object):
    def __init__(self, x, y, theta, grid, config, sense_noise=None):
        # initialize robot pose
        self.x = x
        self.y = y
        self.theta = theta
        self.trajectory = []

        # map that robot navigates in
        # for particles, it is a map with prior probability
        self.grid = grid
        self.grid_size = self.grid.shape

        # probability for updating occupancy map
        self.prior_prob = config['prior_prob']
        self.occupy_prob = config['occupy_prob']
        self.free_prob = config['free_prob']

        # sensing noise for trun robot measurement
        self.sense_noise = sense_noise if sense_noise is not None else 0.0

        # parameters for beam range sensor
        self.num_sensors = config['num_sensors']
        self.radar_theta = np.arange(0, self.num_sensors) * (2 * np.pi / self.num_sensors) + np.pi / self.num_sensors
        self.radar_length = config['radar_length']
        self.radar_range = config['radar_range']

    def set_states(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def get_state(self):
        return (self.x, self.y, self.theta)
    
    def update_trajectory(self):
        self.trajectory.append([self.x, self.y])

    def move(self, turn, forward):
        self.theta = self.theta + turn
        self.theta = wrapAngle(self.theta)

        self.x = self.x + forward * np.cos(self.theta)
        self.y = self.y + forward * np.sin(self.theta)

    def sense(self, world_grid=None):
        measurements, free_grid, occupy_grid = self.ray_casting(world_grid)
        measurements = np.clip(measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors), 0.0, self.radar_range)
        
        return measurements, free_grid, occupy_grid

    def build_radar_beams(self):
        radar_src = np.array([[self.x] * self.num_sensors, [self.y] * self.num_sensors])
        radar_theta = self.radar_theta + self.theta
        radar_rel_dest = np.stack(
            (
                np.cos(radar_theta) * self.radar_length,
                np.sin(radar_theta) * self.radar_length
            ), axis=0
        )

        radar_dest = radar_rel_dest + radar_src

        beams = [None] * self.num_sensors
        for i in range(self.num_sensors):
            x1, y1 = radar_src[:, i]
            x2, y2 = radar_dest[:, i]
            beams[i] = bresenham(x1, y1, x2, y2, self.grid_size[0], self.grid_size[1])

        return beams
    
    def ray_casting(self, world_grid=None):
        beams = self.build_radar_beams()

        loc = np.array([self.x, self.y])
        measurements = [self.radar_range] * self.num_sensors
        free_grid, occupy_grid = [], []

        for i, beam in enumerate(beams):
            dist = np.linalg.norm(beam - loc, axis=1)
            beam = np.array(beam)

            obstacle_position = np.where(self.grid[beam[:, 1], beam[:, 0]] >= 0.9)[0]
            if len(obstacle_position) > 0:
                idx = obstacle_position[0]
                occupy_grid.append(list(beam[idx]))
                free_grid.extend(list(beam[:idx]))
                measurements[i] = dist[idx]
            else:
                free_grid.extend(list(beam))

        return measurements, free_grid, occupy_grid
    
    def update_occupancy_grid(self, free_grid, occupy_grid):
        mask1 = np.logical_and(0 < free_grid[:, 0], free_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < free_grid[:, 1], free_grid[:, 1] < self.grid_size[0])
        free_grid = free_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(False)
        l = prob2logodds(self.grid[free_grid[:, 1], free_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[free_grid[:, 1], free_grid[:, 0]] = logodds2prob(l)

        mask1 = np.logical_and(0 < occupy_grid[:, 0], occupy_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < occupy_grid[:, 1], occupy_grid[:, 1] < self.grid_size[0])
        occupy_grid = occupy_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(True)
        l = prob2logodds(self.grid[occupy_grid[:, 1], occupy_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[occupy_grid[:, 1], occupy_grid[:, 0]] = logodds2prob(l)
    
    def inverse_sensing_model(self, occupy):
        if occupy:
            return self.occupy_prob
        else:
            return self.free_prob
        
    def wall_follow(self, desired_distance=1.0, min_speed=1.0, max_speed=5.0, kp=0.5):
        """
        Adjusts the robot's motion to follow walls at a specified distance.
        Uses a proportional controller to determine the turn rate based on 
        the difference between left and right sensor readings.

        :param desired_distance: Desired distance to keep from the wall (default: 1.0 meter)
        :param min_speed: minimum forward speed (default: 5.0)
        :param max_speed: maximum forward speed (default: 5.0)
        :param kp: Proportional control gain for turning (default: 0.5)

        :return forward: forward control input
        :return turn: turn control input
        """
        # Get sensor readings (z_star) using the sense method
        # z_star, _, _ = self.sense()
        # Define the number of sensors used for each direction
        # sensors_per_direction = self.num_sensors // 4
        # # Group the sensor readings into four directions (front, right, back, left)
        # front = z_star[0:sensors_per_direction]
        # right = z_star[sensors_per_direction:2 * sensors_per_direction]
        # back  = z_star[2 * sensors_per_direction:3 * sensors_per_direction]
        # left  = z_star[3 * sensors_per_direction:self.num_sensors]
        # # Compute the average distance for each direction
        # avg_front = np.mean(front)
        # avg_left = np.mean(left)
        # avg_right = np.mean(right)
        # avg_back = np.mean(back)
        # # Use left sensors for wall following
        # left_error = desired_distance - avg_left
        # # Proportional control to calculate the turning rate
        # turn = kp * (left_error)
        # # Check if there's an obstacle in front
        # if avg_front < 20:  # Adjust threshold based on robot's sensing range
        #     forward_speed = 1  # Slow down if an obstacle is close
        # else:
        #     forward_speed = forward_speed  # Normal speed
        # self.move(turn=turn, forward=forward_speed)
        # print(f"Avg Front: {avg_front:.2f}, Avg Right: {avg_right:.2f}, Avg Back: {avg_back:.2f}, Avg Left: {avg_left:.2f}")
        # Optional: Print averaged sensor data for debugging
        # print(f"Wall Follow - Forward: {forward_command}, Turn: {turn_rate}")

        ## version 2
        z, _, _ = self.sense()  # get sensor data
        num_groups = 4
        sensors_per_direction = self.num_sensors // num_groups
        left_avg  = np.mean(z[0:sensors_per_direction])                             # -135 to -45 degrees
        back_avg  = np.mean(z[sensors_per_direction:2 * sensors_per_direction])     # -45 to 45 degrees
        right_avg = np.mean(z[2 * sensors_per_direction:3 * sensors_per_direction]) # 45 to 135 degrees
        front_avg = np.mean(z[3 * sensors_per_direction:])                          # 135 to 225 degrees
        # l = z[sensors_per_direction * 0.5]  # left sensor
        # b = z[sensors_per_direction * 1.5]  # back sensor
        # r = z[sensors_per_direction * 2.5]  # right sensor
        # f = z[sensors_per_direction * 3.5]  # front sensor
        error = desired_distance - left_avg
        turn = kp * error  # Proportional control for turning
        forward = max_speed - kp * abs(turn)
        forward = max(min_speed, min(forward, max_speed))
        # Adjust for obstacles detected in front
        if front_avg < desired_distance:
            # If an obstacle is detected, stop and turn
            turn += kp * (desired_distance - front_avg)  # Increase turn to avoid collision
            forward = min_speed  # Stop or slow down
        # If the left wall is too far away, turn toward it
        elif left_avg > desired_distance + 5:  # Allow for some tolerance
            turn += kp * (desired_distance + 5 - left_avg)  # Adjust turn to move closer to the wall
        # If the left wall is too close, turn away
        elif left_avg < desired_distance - 5:  # Allow for some tolerance
            turn += kp * (desired_distance - 5 - left_avg)  # Adjust turn to move away from the wall
        # print(f"Front: {front_avg}, right: {right_avg}, Back: {left_avg}, Left: {back_avg}")
        print(f"Wall Following: Forward {forward:.2f}, Turn {turn:.2f}")
        self.move(turn,forward)
        return (turn, forward)
