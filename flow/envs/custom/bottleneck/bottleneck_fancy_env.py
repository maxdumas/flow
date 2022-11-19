import numpy as np
from gym.spaces.box import Box

from flow.controllers import (
    ContinuousRouter,
    IDMController_AvoidAVClumping,
    LaneChangeController_AvoidAVClumping,
    RLController,
    SimLaneChangeController,
)
from flow.core.custom_rewards import fancy_reward_2
from flow.core.params import (
    InFlows,
    NetParams,
    SumoCarFollowingParams,
    SumoLaneChangeParams,
    VehicleParams,
)
from flow.core.state_fragments import get_surrounding_headways
from flow.envs import MultiAgentBottleneckEnv

MAX_LANES = 4  # base number of largest number of lanes in the network
EDGE_LIST = ["1", "2", "3", "4", "5"]  # Edge 1 is before the toll booth
EDGE_BEFORE_TOLL = "1"  # Specifies which edge number is before toll booth
TB_TL_ID = "2"
EDGE_AFTER_TOLL = "2"  # Specifies which edge number is after toll booth
NUM_TOLL_LANES = MAX_LANES

TOLL_BOOTH_AREA = 10  # how far into the edge lane changing is disabled
RED_LIGHT_DIST = 50  # how close for the ramp meter to start going off

EDGE_BEFORE_RAMP_METER = "2"  # Specifies which edge is before ramp meter
EDGE_AFTER_RAMP_METER = "3"  # Specifies which edge is after ramp meter
NUM_RAMP_METERS = MAX_LANES

RAMP_METER_AREA = 80  # Area occupied by ramp meter

MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK = 3  # Average waiting time at fast track
MEAN_NUM_SECONDS_WAIT_AT_TOLL = 15  # Average waiting time at toll

BOTTLE_NECK_LEN = 280  # Length of bottleneck
NUM_VEHICLE_NORM = 20


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # whether the toll booth should be active
    "disable_tb": True,
    # whether the ramp meter is active
    "disable_ramp_metering": True,
}

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,
    # if an RL vehicle exits, place it back at the front
    "add_rl_if_exit": True,
}

# Keys for VSL style experiments
ADDITIONAL_VSL_ENV_PARAMS = {
    # number of controlled regions for velocity bottleneck controller
    "controlled_segments": [
        ("1", 1, True),
        ("2", 1, True),
        ("3", 1, True),
        ("4", 1, True),
        ("5", 1, True),
    ],
    # whether lanes in a segment have the same action or not
    "symmetric": False,
    # which edges are observed
    "observed_segments": [("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1)],
    # whether the inflow should be reset on each rollout
    "reset_inflow": False,
    # the range of inflows to reset on
    "inflow_range": [1000, 2000],
}

START_RECORD_TIME = 0.0  # Time to start recording
PERIOD = 10.0


class MultiAgentBottleneckDesiredThroughputEnv_Fancy(MultiAgentBottleneckEnv):
    """BottleneckDesiredVelocityEnv.

    Environment used to train vehicles to effectively pass through a
    bottleneck by specifying the velocity that RL vehicles should attempt to
    travel in certain regions of space.

    States
        An observation is the number of vehicles in each lane in each
        segment

    Actions
        The action space consist of a list in which each element
        corresponds to the desired speed that RL vehicles should travel in
        that region of space

    Rewards
        The reward is the outflow of the bottleneck plus a reward
        for RL vehicles making forward progress
    """

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        """Initialize BottleneckDesiredVelocityEnv."""
        super().__init__(env_params, sim_params, network, simulator)
        for p in ADDITIONAL_VSL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.format(p))

        # default (edge, segment, controlled) status
        add_env_params = self.env_params.additional_params
        default = [(str(i), 1, True) for i in range(1, 6)]
        super(MultiAgentBottleneckDesiredThroughputEnv_Fancy, self).__init__(
            env_params, sim_params, network
        )
        self.segments = add_env_params.get("controlled_segments", default)

        # number of segments for each edge
        self.num_segments = [segment[1] for segment in self.segments]

        # whether an edge is controlled
        self.is_controlled = [segment[2] for segment in self.segments]

        self.num_controlled_segments = [
            segment[1] for segment in self.segments if segment[2]
        ]

        # sum of segments
        self.total_segments = int(np.sum([segment[1] for segment in self.segments]))
        # sum of controlled segments
        segment_list = [segment[1] for segment in self.segments if segment[2]]
        self.total_controlled_segments = int(np.sum(segment_list))

        # list of controlled edges for comparison
        self.controlled_edges = [segment[0] for segment in self.segments if segment[2]]

        additional_params = env_params.additional_params

        # for convenience, construct the relevant positions defining
        # segments within edges
        # self.slices is a dictionary mapping
        # edge (str) -> segment start location (list of int)
        self.slices = {}
        for edge, num_segments, _ in self.segments:
            edge_length = self.k.network.edge_length(edge)
            self.slices[edge] = np.linspace(0, edge_length, num_segments + 1)

        # get info for observed segments
        self.obs_segments = additional_params.get("observed_segments", [])

        # number of segments for each edge
        self.num_obs_segments = [segment[1] for segment in self.obs_segments]

        # for convenience, construct the relevant positions defining
        # segments within edges
        # self.slices is a dictionary mapping
        # edge (str) -> segment start location (list of int)
        self.obs_slices = {}
        for edge, num_segments in self.obs_segments:
            edge_length = self.k.network.edge_length(edge)
            self.obs_slices[edge] = np.linspace(0, edge_length, num_segments + 1)

        # self.symmetric is True if all lanes in a segment
        # have same action, else False
        self.symmetric = additional_params.get("symmetric")

        # action index tells us, given an edge and a lane,the offset into
        # rl_actions that we should take.
        self.action_index = [0]
        for i, (edge, segment, controlled) in enumerate(self.segments[:-1]):
            if self.symmetric:
                self.action_index += [self.action_index[i] + segment * controlled]
            else:
                num_lanes = self.k.network.num_lanes(edge)
                self.action_index += [
                    self.action_index[i] + segment * controlled * num_lanes
                ]

        self.action_index = {}
        action_list = [0]
        index = 0
        for (edge, num_segments, controlled) in self.segments:
            if controlled:
                if self.symmetric:
                    self.action_index[edge] = [action_list[index]]
                    action_list += [action_list[index] + controlled]
                else:
                    num_lanes = self.k.network.num_lanes(edge)
                    self.action_index[edge] = [action_list[index]]
                    action_list += [
                        action_list[index] + num_segments * controlled * num_lanes
                    ]
                index += 1

    @property
    def observation_space(self):
        """See class definition."""
        num_obs = 0
        # density and velocity for rl and non-rl vehicles per segment
        # Last element is the outflow
        for segment in self.obs_segments:
            num_obs += 4 * segment[1] * self.k.network.num_lanes(segment[0])
        # outflow
        num_obs += 1
        # individual obs
        num_obs += 12
        return Box(
            low=-float("inf"), high=float("inf"), shape=(num_obs,), dtype=np.float32
        )

    @property
    def action_space(self):
        """See class definition."""
        if self.symmetric:
            action_size = self.total_controlled_segments
        else:
            action_size = 0.0
            for segment in self.segments:  # iterate over segments
                if segment[2]:  # if controlled
                    num_lanes = self.k.network.num_lanes(segment[0])
                    action_size += num_lanes * segment[1]
        add_params = self.env_params.additional_params
        max_accel = add_params.get("max_accel")
        max_decel = add_params.get("max_decel")
        return Box(
            low=-max_decel * self.sim_step,
            high=max_accel * self.sim_step,
            shape=(int(action_size) + 3,),
            dtype=np.float32,
        )

    def get_state(self):
        """Return aggregate statistics of different segments of the bottleneck.

        The state space of the system is defined by splitting the bottleneck up
        into edges and then segments in each edge. The class variable
        self.num_obs_segments specifies how many segments each edge is cut up
        into. Each lane defines a unique segment: we refer to this as a
        lane-segment. For example, if edge 1 has four lanes and three segments,
        then we have a total of 12 lane-segments. We will track the aggregate
        statistics of the vehicles in each lane segment.

        For each lane-segment we return the:

        * Number of vehicles on that segment.
        * Number of AVs (referred to here as rl_vehicles) in the segment.
        * The average speed of the vehicles in that segment.
        * The average speed of the rl vehicles in that segment.

        Finally, we also append the total outflow of the bottleneck over the
        last 20 * self.sim_step seconds.
        """

        # General global observation
        num_vehicles_list = []
        num_rl_vehicles_list = []
        vehicle_speeds_list = []
        rl_speeds_list = []
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
                segment = np.searchsorted(self.obs_slices[edge], pos_list[i]) - 1
                if id in self.k.vehicle.get_rl_ids():
                    rl_vehicle_speeds[
                        segment, lane_list[i]
                    ] += self.k.vehicle.get_speed(id)
                    num_rl_vehicles[segment, lane_list[i]] += 1
                else:
                    vehicle_speeds[segment, lane_list[i]] += self.k.vehicle.get_speed(
                        id
                    )
                    num_vehicles[segment, lane_list[i]] += 1

            # normalize

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
        mean_speed = np.nan_to_num(
            [
                vehicle_speeds_list[i] / unnorm_veh_list[i]
                if int(unnorm_veh_list[i])
                else 0
                for i in range(num_veh)
            ]
        )
        mean_speed_norm = mean_speed / 50
        mean_rl_speed = (
            np.nan_to_num(
                [
                    rl_speeds_list[i] / unnorm_rl_list[i]
                    if int(unnorm_rl_list[i])
                    else 0
                    for i in range(num_rl)
                ]
            )
            / 50
        )
        inflow = self.k.vehicle.get_inflow_rate(10 * self.sim_step)
        inflow = inflow if inflow != 0.0 else 2000.0
        outflow = np.asarray(
            self.k.vehicle.get_outflow_rate(10 * self.sim_step) / inflow
        )
        global_obs = np.concatenate(
            (
                num_vehicles_list,
                num_rl_vehicles_list,
                mean_speed_norm,
                mean_rl_speed,
                [outflow],
            )
        )

        # observation for each AV
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

            obs.update({rl_id: np.concatenate((global_obs, observation))})
            # obs.update({rl_id: global_obs})
            # obs.update({rl_id: observation})
        return obs

    def _apply_rl_actions(self, rl_actions):
        """
        RL actions are split up into 3 levels.

        * First, they're split into edge actions.
        * Then they're split into segment actions.
        * Then they're split into lane actions.
        """
        if rl_actions:
            for rl_id in self.k.vehicle.get_rl_ids():
                actions = rl_actions[rl_id]
                edge = self.k.vehicle.get_edge(rl_id)
                lane = self.k.vehicle.get_lane(rl_id)
                if edge:
                    # If in outer lanes, on a controlled edge, in a controlled lane
                    if edge[0] != ":" and edge in self.controlled_edges:
                        pos = self.k.vehicle.get_position(rl_id)

                        if not self.symmetric:
                            num_lanes = self.k.network.num_lanes(edge)
                            # find what segment we fall into
                            bucket = np.searchsorted(self.slices[edge], pos) - 1
                            action = actions[
                                int(lane) + bucket * num_lanes + self.action_index[edge]
                            ]
                        else:
                            # find what segment we fall into
                            bucket = np.searchsorted(self.slices[edge], pos) - 1
                            action = actions[bucket + self.action_index[edge]]

                        max_speed_curr = self.k.vehicle.get_max_speed(rl_id)
                        next_max = np.clip(max_speed_curr + action, 0.01, 23.0)
                        self.k.vehicle.set_max_speed(rl_id, next_max)

                        lane_change_duration = self.env_params.additional_params[
                            "lane_change_duration"
                        ]
                        # duration in sim steps
                        duration_sim_step = lane_change_duration / self.sim_step
                        if (
                            self.time_counter
                            <= duration_sim_step + self.k.vehicle.get_last_lc(rl_id)
                        ):
                            lane_change_action = 0
                            # print(self.time_counter, self.k.vehicle.get_last_lc(rl_id), rl_id, "yes")
                        else:
                            lane_change_softmax = np.exp(actions[-3:])
                            lane_change_softmax /= np.sum(lane_change_softmax)
                            lane_change_action = np.random.choice(
                                [-1, 0, 1], p=lane_change_softmax
                            )

                        self.k.vehicle.apply_lane_change(rl_id, lane_change_action)

                    else:
                        # set the desired velocity of the controller to the default
                        self.k.vehicle.set_max_speed(rl_id, 23.0)

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        if self.env_params.evaluate:
            if self.time_counter == self.env_params.horizon:
                return self.k.vehicle.get_outflow_rate(500)
            return 0.0

        if rl_actions is None:
            return {}

        if kwargs["fail"]:
            return 0.0

        num_rl = len(self.k.vehicle.get_rl_ids())
        return {
            rl_id: fancy_reward_2(self, rl_id) / num_rl
            for rl_id in self.k.vehicle.get_rl_ids()
        }

    def reset(self):
        """Reset the environment with a new inflow rate.

        The diverse set of inflows are used to generate a policy that is more
        robust with respect to the inflow rate. The inflow rate is update by
        creating a new network similar to the previous one, but with a new
        Inflow object with a rate within the additional environment parameter
        "inflow_range", which is a list consisting of the smallest and largest
        allowable inflow rates.

        **WARNING**: The inflows assume there are vehicles of type
        "followerstopper" and "human" within the VehicleParams object.
        """
        add_params = self.env_params.additional_params
        if add_params.get("reset_inflow"):
            inflow_range = add_params.get("inflow_range")
            flow_rate = (
                np.random.uniform(min(inflow_range), max(inflow_range)) * self.scaling
            )

            # We try this for 100 trials in case unexpected errors during
            # instantiation.
            for _ in range(100):
                try:
                    # introduce new inflows within the pre-defined inflow range
                    inflow = InFlows()
                    inflow.add(
                        veh_type="rl",  # FIXME: make generic
                        edge="1",
                        vehs_per_hour=flow_rate * 0.25,
                        depart_lane="random",
                        depart_speed=10,
                    )
                    inflow.add(
                        veh_type="idm",
                        edge="1",
                        vehs_per_hour=flow_rate * 0.75,
                        depart_lane="random",
                        depart_speed=10,
                    )

                    # all other network parameters should match the previous
                    # environment (we only want to change the inflow)
                    additional_net_params = {
                        "scaling": self.scaling,
                        "speed_limit": self.net_params.additional_params["speed_limit"],
                    }
                    net_params = NetParams(
                        inflows=inflow, additional_params=additional_net_params
                    )

                    vehicles = VehicleParams()
                    vehicles.add(
                        veh_id="idm",
                        acceleration_controller=(
                            IDMController_AvoidAVClumping,
                            {"noise": 0.2},
                        ),
                        lane_change_controller=(
                            LaneChangeController_AvoidAVClumping,
                            {},
                        ),
                        car_following_params=SumoCarFollowingParams(
                            speed_mode="all_checks",
                        ),
                        lane_change_params=SumoLaneChangeParams(
                            lane_change_mode="sumo_default",
                        ),
                        num_vehicles=1 * self.scaling,
                    )
                    vehicles.add(
                        veh_id="rl",
                        acceleration_controller=(RLController, {}),
                        lane_change_controller=(SimLaneChangeController, {}),
                        routing_controller=(ContinuousRouter, {}),
                        car_following_params=SumoCarFollowingParams(
                            speed_mode="all_checks",
                        ),
                        lane_change_params=SumoLaneChangeParams(
                            lane_change_mode="sumo_default",
                        ),
                        num_vehicles=1 * self.scaling,
                    )

                    # recreate the network object
                    self.network = self.network.__class__(
                        name=self.network.orig_name,
                        vehicles=vehicles,
                        net_params=net_params,
                        initial_config=self.initial_config,
                        traffic_lights=self.network.traffic_lights,
                    )
                    observation = super().reset()

                    # reset the timer to zero
                    self.time_counter = 0

                    return observation

                except Exception as e:
                    print("error on reset ", e)

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation
