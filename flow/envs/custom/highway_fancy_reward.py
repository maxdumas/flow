"""Environment used to train vehicles to improve traffic on a highway."""
import numpy as np
from gym.spaces.box import Box
from flow.core.custom_rewards import fancy_reward

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


class MultiAgentHighwayFancyEnv(MultiEnv):
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

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(-float("inf"), float("inf"), shape=(5,), dtype=np.float32)

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
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(lead_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            observation = np.array(
                [
                    this_speed / max_speed,
                    (lead_speed - this_speed) / max_speed,
                    lead_head / max_length,
                    (this_speed - follow_speed) / max_speed,
                    follow_head / max_length,
                ]
            )

            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if self.env_params.evaluate:
                # reward is speed of vehicle if we are in evaluation mode
                reward = self.k.vehicle.get_speed(rl_id)
            elif kwargs["fail"]:
                # reward is 0 if a collision occurred
                reward = 0
            else:
                dists_to_leader = 0
                dists_to_follower = 0
                min_dist_to_leader = 0
                min_dist_to_follower = 0

                # penalize small headways to the leader of this AV
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] and self.k.vehicle.get_speed(rl_id) > 0:
                    # smallest acceptable distance headway
                    min_dist_to_leader = self.k.vehicle.get_distance_preference(lead_id)
                    # distance to leader
                    dists_to_leader = max(self.k.vehicle.get_headway(rl_id), 0)
                else:
                    # if there is no leader
                    dists_to_leader = -1

                # penalize small headways for the follower of this AV
                follower_id = self.k.vehicle.get_follower(rl_id)
                if (
                    follower_id not in ["", None]
                    and self.k.vehicle.get_speed(rl_id) > 0
                ):
                    # smallest acceptable distance headway
                    min_dist_to_follower = self.k.vehicle.get_distance_preference(
                        follower_id
                    )
                    # distance to follower
                    dists_to_follower = max(self.k.vehicle.get_headway(follower_id), 0)
                else:
                    # if there is no follower
                    dists_to_follower = -1

                reward = fancy_reward(
                    self,
                    dists_to_leader,
                    dists_to_follower,
                    min_dist_to_leader,
                    min_dist_to_follower,
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
