"""Environment used to train vehicles to improve traffic on a highway."""
import numpy as np
from gym.spaces.box import Box
from flow.core.rewards import desired_velocity
from flow.envs.base import Env


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    "observed_segments": [("1", 5)]
}


class HighwayRegulationEnv(Env):
    """Partially observable multi-agent environment for an highway with ramps.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open highway network.

    The highway can contain an arbitrary number of entrance and exit ramps, and
    is intended to be used with the HighwayRampsNetwork network.

    The policy is shared among the agents, so there can be a non-constant
    number of RL vehicles throughout the simulation.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the speed of the autonomous vehicle.

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters, as well as three values that will be converted
        into probabilities via softmax to decide of a lane change (left, none
        or right).

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity specified in the environment parameters, while
        slightly penalizing small time headways among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.segments = [("highway_0", 5, True)]

        additional_params = env_params.additional_params
        self.obs_segments = additional_params.get("observed_segments", [])

        # number of segments for each edge
        self.num_obs_segments = [segment[1] for segment in self.obs_segments]

        self.obs_slices = {}
        for edge, num_segments in self.obs_segments:
            edge_length = self.k.network.edge_length(edge)
            self.obs_slices[edge] = np.linspace(0, edge_length,
                                                num_segments + 1)

        self.slices = {}
        for edge, num_segments, _ in self.segments:
            edge_length = self.k.network.edge_length(edge)
            self.slices[edge] = np.linspace(0, edge_length, num_segments + 1)


        self.action_index = {}
        action_list = [0]
        index = 0
        for (edge, num_segments, controlled) in self.segments:
            if controlled:
                num_lanes = self.k.network.num_lanes(edge)
                self.action_index[edge] = [action_list[index]]
                action_list += [
                    action_list[index] +
                    num_segments * controlled * num_lanes
                ]
                index += 1


    @property
    def observation_space(self):
        """See class definition."""
        # return Box(-float('inf'), float('inf'), shape=(5,), dtype=np.float32)
        num_obs = 0
        # density and velocity for rl and non-rl vehicles per segment
        # Last element is the outflow
        for segment in self.obs_segments:
            num_obs += 4 * segment[1] * self.k.network.num_lanes(segment[0])
        num_obs += 1
        return Box(low=-float("inf"), high=float("inf"), shape=(num_obs, ), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        action_size = 0.0
        for segment in self.segments:  # iterate over segments
            if segment[2]:  # if controlled
                num_lanes = self.k.network.num_lanes(segment[0])
                action_size += num_lanes * segment[1]
        add_params = self.env_params.additional_params
        max_accel = add_params.get("max_accel")
        max_decel = add_params.get("max_decel")
        return Box(
            low=-max_decel*self.sim_step, high=max_accel*self.sim_step,
            shape=(int(action_size), ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        for rl_id in self.k.vehicle.get_rl_ids():
            edge = self.k.vehicle.get_edge(rl_id)
            lane = self.k.vehicle.get_lane(rl_id)
            if edge:
                pos = self.k.vehicle.get_position(rl_id)

                num_lanes = self.k.network.num_lanes(edge)
                # find what segment we fall into
                bucket = np.searchsorted(self.slices[edge], pos) - 1
                action = rl_actions[int(lane) + bucket * num_lanes +
                                        self.action_index[edge]]

                max_speed_curr = self.k.vehicle.get_max_speed(rl_id)
                next_max = np.clip(max_speed_curr + action, 0.01, 23.0)
                self.k.vehicle.set_max_speed(rl_id, next_max)

    def get_state(self):
        """See class definition."""
        num_vehicles_list = []
        num_rl_vehicles_list = []
        vehicle_speeds_list = []
        rl_speeds_list = []
        EDGE_LIST = ["highway_0"] 
        for i, edge in enumerate(EDGE_LIST):
            num_lanes = self.k.network.num_lanes(edge)
            num_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            num_rl_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            rl_vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            ids = self.k.vehicle.get_ids_by_edge(edge)
            lane_list = self.k.vehicle.get_lane(ids)
            pos_list = self.k.vehicle.get_position(ids)
            for i, id in enumerate(ids):
                segment = np.searchsorted(self.obs_slices[edge],
                                          pos_list[i]) - 1
                if id in self.k.vehicle.get_rl_ids():
                    rl_vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_rl_vehicles[segment, lane_list[i]] += 1
                else:
                    vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_vehicles[segment, lane_list[i]] += 1

            NUM_VEHICLE_NORM = 20
            num_vehicles /= NUM_VEHICLE_NORM
            num_rl_vehicles /= NUM_VEHICLE_NORM
            num_vehicles_list += num_vehicles.flatten().tolist()
            num_rl_vehicles_list += num_rl_vehicles.flatten().tolist()
            vehicle_speeds_list += vehicle_speeds.flatten().tolist()
            rl_speeds_list += rl_vehicle_speeds.flatten().tolist()

        unnorm_veh_list = np.asarray(num_vehicles_list) * NUM_VEHICLE_NORM
        unnorm_rl_list = np.asarray(num_rl_vehicles_list) * NUM_VEHICLE_NORM

        # compute the mean speed if the speed isn't zero
        num_rl = len(num_rl_vehicles_list)
        num_veh = len(num_vehicles_list)
        mean_speed = np.nan_to_num([
            vehicle_speeds_list[i] / unnorm_veh_list[i]
            if int(unnorm_veh_list[i]) else 0 for i in range(num_veh)
        ])
        mean_speed_norm = mean_speed / 50
        mean_rl_speed = np.nan_to_num([
            rl_speeds_list[i] / unnorm_rl_list[i]
            if int(unnorm_rl_list[i]) else 0 for i in range(num_rl)
        ]) / 50
        outflow = np.asarray(
            self.k.vehicle.get_outflow_rate(20 * self.sim_step) / 2000.0)
        return np.concatenate((num_vehicles_list, num_rl_vehicles_list,
                               mean_speed_norm, mean_rl_speed, [outflow]))

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        # vel = np.array(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))

        # if any(vel < -100):
        #     return 0.
        # if len(vel) == 0:
        #     return 0.

        # return np.mean(vel) / 23.0
        reward = self.k.vehicle.get_outflow_rate(10 * self.sim_step) / 2000.0
        return reward


    # def additional_command(self):
    #     """See parent class.

    #     Define which vehicles are observed for visualization purposes.
    #     """
    #     # specify observed vehicles
    #     for rl_id in self.k.vehicle.get_rl_ids():
    #         # leader
    #         lead_id = self.k.vehicle.get_leader(rl_id)
    #         if lead_id:
    #             self.k.vehicle.set_observed(lead_id)
    #         # follower
    #         follow_id = self.k.vehicle.get_follower(rl_id)
    #         if follow_id:
    #             self.k.vehicle.set_observed(follow_id)
