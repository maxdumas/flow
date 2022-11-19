"""Environment used to train vehicles to improve traffic on a highway."""
import numpy as np
from gym.spaces.box import Box
from flow.core.custom_rewards import fancy_reward
from flow.core.state_fragments import get_surrounding_headways

from flow.envs.multiagent.base import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    "max_accel": 1,
    # maximum deceleration of autonomous vehicles
    "max_decel": 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 3,
}


class MultiAgentFigureEightFancyEnv(MultiEnv):
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
        The observation consists of:
            * the speeds and bumper-to-bumper headways of the vehicles immediately preceding and following autonomous vehicle,
            * whether or not the leader and follower are also autonomous vehicles
            * the speed of the autonomous vehicle
            * the position of the autonomous vehicle in the network
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

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(-5, 5, shape=(12,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(4,),  # (4,),
            dtype=np.float32,
        )

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            # duration in seconds
            lane_change_duration = self.env_params.additional_params[
                "lane_change_duration"
            ]
            # duration in sim steps
            duration_sim_step = lane_change_duration / self.sim_step
            for rl_id, actions in rl_actions.items():
                if rl_id in self.k.vehicle.get_rl_ids():
                    accel = actions[0]

                    if (
                        self.time_counter
                        <= duration_sim_step + self.k.vehicle.get_last_lc(rl_id)
                    ):
                        lane_change_action = 0
                        # print(self.time_counter, self.k.vehicle.get_last_lc(rl_id), rl_id, "yes")
                    else:
                        lane_change_softmax = np.exp(actions[1:4])
                        lane_change_softmax /= np.sum(lane_change_softmax)
                        lane_change_action = np.random.choice(
                            [-1, 0, 1], p=lane_change_softmax
                        )

                    self.k.vehicle.apply_acceleration(rl_id, accel)
                    self.k.vehicle.apply_lane_change(rl_id, lane_change_action)

    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.vehicle.get_x_by_id(rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                leader_is_av = 0.0
                lead_speed = max_speed
            else:
                leader_is_av = float(lead_id in self.k.vehicle.get_rl_ids())
                lead_speed = self.k.vehicle.get_speed(lead_id)

            if follower in ["", None]:
                # in case follower is not visible
                follower_is_av = 0.0
                follow_speed = 0
            else:
                follower_is_av = float(follower in self.k.vehicle.get_rl_ids())
                follow_speed = self.k.vehicle.get_speed(follower)

            observation = np.array(
                [
                    this_pos / max_length,
                    this_speed / max_speed,
                    (lead_speed - this_speed) / max_speed,
                    (this_speed - follow_speed) / max_speed,
                    leader_is_av,  # This is 1.0 if the leader is also an AV, 0.0 otherwise.
                    follower_is_av,  # This is 1.0 if the follower is also an AV, 0.0 otherwise.
                    *get_surrounding_headways(self, rl_id),
                ]
            )

            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        max_length = self.k.network.length()

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if self.env_params.evaluate:
                # reward is speed of vehicle if we are in evaluation mode
                reward = self.k.vehicle.get_speed(rl_id)
            elif kwargs["fail"]:
                # reward is 0 if a collision occurred
                reward = 0.0
            else:
                headways_normalized = get_surrounding_headways(self, rl_id)

                lead_id = self.k.vehicle.get_leader(rl_id)
                min_dist_to_leader_normalized = (
                    self.k.vehicle.get_distance_preference(lead_id) / max_length
                )

                follower_id = self.k.vehicle.get_follower(rl_id)
                min_dist_to_follower_normalized = (
                    self.k.vehicle.get_distance_preference(follower_id) / max_length
                )

                reward = fancy_reward(
                    self,
                    headways_normalized[1],
                    headways_normalized[4],
                    min_dist_to_leader_normalized,
                    min_dist_to_follower_normalized,
                    fail=kwargs["fail"],
                )
            rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.
        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        return super().reset()
