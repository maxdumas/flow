"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.base import Env

from gym.spaces.box import Box

from copy import deepcopy
import numpy as np
import random
from scipy.optimize import fsolve

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


# def v_eq_max_function(v, *args):
#     """Return the error between the desired and actual equivalent gap."""
#     num_vehicles, length = args

#     # maximum gap in the presence of one rl vehicle
#     s_eq_max = (length - num_vehicles * 5) / (num_vehicles - 1)

#     v0 = 30
#     s0 = 2
#     tau = 1
#     gamma = 4

#     error = s_eq_max - (s0 + v * tau) * (1 - (v / v0) ** gamma) ** -0.5

#     return error


class RegulationEnv(Env):
    """Fully observable wave attenuation environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in a variable density ring road.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on. If set to None, the environment sticks to the ring
      road specified in the original network definition.

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function rewards high average speeds from all vehicles in
        the network, and penalizes accelerations by the rl vehicle.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)


    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * self.initial_vehicles.num_rl_vehicles
        ub = [max_accel, 1] * self.initial_vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ["Velocity", "Absolute_pos", "Lane"]
        return Box(
            low=0,
            high=1.5,
            shape=(3 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""

        num_rl = self.k.vehicle.num_rl_vehicles
        acceleration = rl_actions[::2][:num_rl]
        direction = np.round(rl_actions[1::2])[:num_rl]

        # self.k.vehicle.apply_acceleration(self.k.vehicle.get_rl_ids(), acc=acceleration)
        self.k.vehicle.apply_lane_change(
            self.k.vehicle.get_rl_ids(), direction=direction)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        # the get_ids() method is used to get the names of all vehicles in the network
        if rl_actions is None:
            return 0
            
        # ids = self.k.vehicle.get_ids()

        # # we next get a list of the speeds of all vehicles in the network
        # speeds = self.k.vehicle.get_speed(ids)

        # num_rl = self.k.vehicle.num_rl_vehicles
        # lanes = self.k.vehicle.get_lane(ids)
        # acceleration = rl_actions[::2][:num_rl]
        # direction = np.round(rl_actions[1::2])[:num_rl]

        human_ids = self.k.vehicle.get_human_ids()
        human_speeds = [self.k.vehicle.get_speed(veh_id)
                 for veh_id in human_ids]

        rl_ids = self.k.vehicle.get_rl_ids()
        rl_speeds = [self.k.vehicle.get_speed(veh_id)
                 for veh_id in rl_ids]

        # reward = np.sum(rl_speeds)
        # Punish if human cars is faster than rl cars
        reward = len(human_ids) * np.sum(rl_speeds) - np.sum(human_speeds)


        # finally, we return the average of all these speeds as the reward
        return reward

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        ids = self.k.vehicle.get_ids()

        speed = [self.k.vehicle.get_speed(veh_id) / max_speed
                 for veh_id in ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / length
               for veh_id in ids]
        lane = [self.k.vehicle.get_lane(veh_id) / max_lanes
                for veh_id in ids]

        return np.array(speed + pos + lane)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

    # def reset(self):
    #     """See parent class.

    #     The sumo instance is reset with a new ring length, and a number of
    #     steps are performed with the rl vehicle acting as a human vehicle.
    #     """
    #     # skip if ring length is None
    #     if self.env_params.additional_params['ring_length'] is None:
    #         return super().reset()

    #     # reset the step counter
    #     self.step_counter = 0

    #     # update the network
    #     initial_config = InitialConfig(bunching=50, min_gap=0)
    #     length = random.randint(
    #         self.env_params.additional_params['ring_length'][0],
    #         self.env_params.additional_params['ring_length'][1])
    #     additional_net_params = {
    #         'length':
    #             length,
    #         'lanes':
    #             self.net_params.additional_params['lanes'],
    #         'speed_limit':
    #             self.net_params.additional_params['speed_limit'],
    #         'resolution':
    #             self.net_params.additional_params['resolution']
    #     }
    #     net_params = NetParams(additional_params=additional_net_params)

    #     self.network = self.network.__class__(
    #         self.network.orig_name, self.network.vehicles,
    #         net_params, initial_config)
    #     self.k.vehicle = deepcopy(self.initial_vehicles)
    #     self.k.vehicle.kernel_api = self.k.kernel_api
    #     self.k.vehicle.master_kernel = self.k

    #     # solve for the velocity upper bound of the ring
    #     v_guess = 4
    #     # v_eq_max = fsolve(v_eq_max_function, np.array(v_guess),
    #     #                   args=(len(self.initial_ids), length))[0]

    #     print('\n-----------------------')
    #     print('ring length:', net_params.additional_params['length'])
    #     # print('v_max:', v_eq_max)
    #     print('-----------------------')

    #     # restart the sumo instance
    #     self.restart_simulation(
    #         sim_params=self.sim_params,
    #         render=self.sim_params.render)

    #     # perform the generic reset function
    #     return super().reset()

