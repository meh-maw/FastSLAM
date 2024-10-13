import numpy as np
from utils import wrapAngle, normalDistribution

class KinematicBicycleModel(object):
    def __init__(self, config):
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.alpha3 = config['alpha3']
        self.alpha4 = config['alpha4']
        self.L = config['wheelbase']  # Wheelbase of the vehicle

    def sample_motion_model(self, prev_odo, curr_odo, prev_pose):
        delta = wrapAngle(np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2])
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        beta = np.arctan(np.tan(delta) / 2)  # Slip angle approximation

        delta = wrapAngle(delta - np.random.normal(0, self.alpha1 * delta ** 2 + self.alpha2 * trans ** 2))
        trans = trans - np.random.normal(0, self.alpha3 * trans ** 2 + self.alpha4 * delta ** 2)

        # Update state using the kinematic bicycle model
        x = prev_pose[0] + trans * np.cos(prev_pose[2] + beta)
        y = prev_pose[1] + trans * np.sin(prev_pose[2] + beta)
        theta = prev_pose[2] + trans * np.sin(beta) / self.L

        return (x, y, wrapAngle(theta))

    def motion_model(self, prev_odo, curr_odo, prev_pose, curr_pose):
        delta = wrapAngle(np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2])
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        beta = np.arctan(np.tan(delta) / 2)

        delta_prime = wrapAngle(np.arctan2(curr_pose[1] - prev_pose[1], curr_pose[0] - prev_pose[0]) - prev_pose[2])
        trans_prime = np.sqrt((curr_pose[0] - prev_pose[0]) ** 2 + (curr_pose[1] - prev_pose[1]) ** 2)
        beta_prime = np.arctan(np.tan(delta_prime) / 2)

        p1 = normalDistribution(wrapAngle(delta - delta_prime), self.alpha1 * delta_prime ** 2 + self.alpha2 * trans_prime ** 2)
        p2 = normalDistribution(trans - trans_prime, self.alpha3 * trans_prime ** 2 + self.alpha4 * delta_prime ** 2)
        p3 = normalDistribution(wrapAngle(beta - beta_prime), self.alpha1 * beta_prime ** 2 + self.alpha2 * trans_prime ** 2)

        return p1 * p2 * p3
