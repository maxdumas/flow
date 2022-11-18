import numpy as np
from gym.spaces.box import Box
from flow.core.custom_rewards import fancy_reward_2

from flow.envs.multiagent.base import MultiEnv
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import InFlows, NetParams
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.envs import BottleneckEnv

from copy import deepcopy

import numpy as np
from gym.spaces.box import Box

from flow.core import rewards
from flow.envs.base import Env

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
    "controlled_segments": [("1", 1, True), ("2", 1, True), ("3", 1, True),
                            ("4", 1, True), ("5", 1, True)],
    # whether lanes in a segment have the same action or not
    "symmetric":
    False,
    # which edges are observed
    "observed_segments": [("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1)],
    # whether the inflow should be reset on each rollout
    "reset_inflow":
    False,
    # the range of inflows to reset on
    "inflow_range": [1000, 2000]
}

START_RECORD_TIME = 0.0  # Time to start recording
PERIOD = 10.0

class MultiAgentBottleneckEnv(MultiEnv):
    """Abstract bottleneck environment.

    This environment is used as a simplified representation of the toll booth
    portion of the bay bridge. Contains ramp meters, and a toll both.

    Additional
    ----------
        Vehicles are rerouted to the start of their original routes once
        they reach the end of the network in order to ensure a constant
        number of vehicles.

    Attributes
    ----------
    scaling : int
        A factor describing how many lanes are in the system. Scaling=1 implies
        4 lanes going to 2 going to 1, scaling=2 implies 8 lanes going to 4
        going to 2, etc.
    edge_dict : dict of dicts
        A dict mapping edges to a dict of lanes where each entry in the lane
        dict tracks the vehicles that are in that lane. Used to save on
        unnecessary lookups.
    cars_waiting_for_toll : {veh_id: {lane_change_mode: int, color: (int)}}
        A dict mapping vehicle ids to a dict tracking the color and lane change
        mode of vehicles before they entered the toll area. When vehicles exit
        the tollbooth area, these values are used to set the lane change mode
        and color of the vehicle back to how they were before they entered the
        toll area.
    cars_before_ramp : {veh_id: {lane_change_mode: int, color: (int)}}
        Identical to cars_waiting_for_toll, but used to track cars approaching
        the ramp meter versus approaching the tollbooth.
    toll_wait_time : np.ndarray(float)
        Random value, sampled from a gaussian indicating how much a vehicle in
        each lane should wait to pass through the toll area. This value is
        re-sampled for each approaching vehicle. That is, whenever a vehicle
        approaches the toll area, we re-sample from the Gaussian to determine
        its weight time.
    fast_track_lanes : np.ndarray(int)
        Middle lanes of the tollbooth are declared fast-track lanes, this numpy
        array keeps track of which lanes these are. At a fast track lane, the
        mean of the Gaussian from which we sample wait times is given by
        MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK.
    tl_state : str
        String tracking the color of the traffic lights at the tollbooth. These
        traffic lights are used imitate the effect of a tollbooth. If lane 1-4
        are respectively green, red, red, green, then this string would be
        "GrrG"
    n_crit : int
        The ALINEA algorithm adjusts the ratio of red to green time for the
        ramp-metering traffic light based on feedback on how congested the
        system is. As the measure of congestion, we use the number of vehicles
        stuck in the bottleneck (edge 4). The critical operating value it tries
        to stabilize the number of vehicles in edge 4 is n_crit. If there are
        more than n_crit vehicles on edge 4, we increase the fraction of red
        time to decrease the inflow to edge 4.
    q_max : float
        The ALINEA algorithm tries to control the flow rate through the ramp
        meter. q_max is the maximum possible estimated flow we allow through
        the bottleneck and can be converted into a maximum value for the ratio
        of green to red time that we allow.
    q_min : float
        Similar to q_max, this is used to set the minimum value of green to red
        ratio that we allow.
    q : float
        This value tracks the flow we intend to allow through the bottleneck.
        For details on how it is computed, please read the alinea method or the
        paper linked in that method.
    feedback_update_time : float
        The parameters of the ALINEA algorithm are only updated every
        feedback_update_time seconds.
    feedback_timer : float
        This keeps track of how many seconds have passed since the ALINEA
        parameters were last updated. If it exceeds feedback_update_time, the
        parameters are updated
    cycle_time : int
        This value tracks how long a green-red cycle of the ramp meter is. The
        first self.green_time seconds will be green and the remainder of the
        cycle will be red.
    ramp_state : np.ndarray
        Array of floats of length equal to the number of lanes. For each lane,
        this value tracks how many seconds of a given cycle have passed in that
        lane. Each lane is offset from its adjacent lanes by
        cycle_offset/(self.scaling * MAX_LANES) seconds. This offsetting means
        that lights are offset in when they releasse vehicles into the
        bottleneck. This helps maximize the throughput of the ramp meter.
    green_time : float
        How many seconds of a given cycle the light should remain green 4.
        Defaults to 4 as this is just enough time for two vehicles to enter the
        bottleneck from a given traffic light.
    feedback_coeff : float
        This is the gain on the feedback in the ALINEA algorithm
    smoothed_num : np.ndarray
        Numpy array keeping track of how many vehicles were in edge 4 over the
        last 10 time seconds. This provides a more stable estimate of the
        number of vehicles in edge 4.
    outflow_index : int
        Keeps track of which index of smoothed_num we should update with the
        latest number of vehicles in the bottleneck. Should eventually be
        deprecated as smoothed_num should be switched to a queue instead of an
        array.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the BottleneckEnv class."""
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        env_add_params = self.env_params.additional_params
        # tells how scaled the number of lanes are
        self.scaling = network.net_params.additional_params.get("scaling", 1)
        self.edge_dict = dict()
        self.cars_waiting_for_toll = dict()
        self.cars_before_ramp = dict()
        self.toll_wait_time = np.abs(
            np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                             4 / self.sim_step, NUM_TOLL_LANES * self.scaling))
        self.fast_track_lanes = range(
            int(np.ceil(1.5 * self.scaling)), int(np.ceil(2.6 * self.scaling)))

        self.tl_state = ""

        # values for the ALINEA ramp meter algorithm
        self.n_crit = env_add_params.get("n_crit", 8)
        self.q_max = env_add_params.get("q_max", 1100)
        self.q_min = env_add_params.get("q_min", .25 * 1100)
        self.q = self.q_min  # ramp meter feedback controller
        self.feedback_update_time = env_add_params.get("feedback_update", 15)
        self.feedback_timer = 0.0
        self.cycle_time = 6
        cycle_offset = 8
        self.ramp_state = np.linspace(0,
                                      cycle_offset * self.scaling * MAX_LANES,
                                      self.scaling * MAX_LANES)
        self.green_time = 4
        self.feedback_coeff = env_add_params.get("feedback_coeff", 20)

        self.smoothed_num = np.zeros(10)  # averaged number of vehs in '4'
        self.outflow_index = 0

    def additional_command(self):
        """Build a dict with vehicle information.

        The dict contains the list of vehicles and their position for each edge
        and for each edge within the edge.
        """
        super().additional_command()

        # build a dict containing the list of vehicles and their position for
        # each edge and for each lane within the edge
        empty_edge = [[] for _ in range(MAX_LANES * self.scaling)]

        self.edge_dict = {k: deepcopy(empty_edge) for k in EDGE_LIST}
        for veh_id in self.k.vehicle.get_ids():
            try:
                edge = self.k.vehicle.get_edge(veh_id)
                if edge not in self.edge_dict:
                    self.edge_dict[edge] = deepcopy(empty_edge)
                lane = self.k.vehicle.get_lane(veh_id)  # integer
                pos = self.k.vehicle.get_position(veh_id)
                self.edge_dict[edge][lane].append((veh_id, pos))
            except Exception:
                pass

        if not self.env_params.additional_params['disable_tb']:
            self.apply_toll_bridge_control()
        if not self.env_params.additional_params['disable_ramp_metering']:
            self.ramp_meter_lane_change_control()
            self.alinea()

        # compute the outflow
        veh_ids = self.k.vehicle.get_ids_by_edge('4')
        self.smoothed_num[self.outflow_index] = len(veh_ids)
        self.outflow_index = \
            (self.outflow_index + 1) % self.smoothed_num.shape[0]

    def ramp_meter_lane_change_control(self):
        """Control lane change behavior of vehicles near the ramp meters.

        If the traffic lights after the toll booth are enabled
        ('disable_ramp_metering' is False), we want to change the lane changing
        behavior of vehicles approaching the lights so that they stop changing
        lanes. This method disables their lane changing before the light and
        re-enables it after they have passed the light.

        Additionally, to visually make it clearer that the lane changing
        behavior of the vehicles has been adjusted, we temporary set the color
        of the affected vehicles to light blue.
        """
        cars_that_have_left = []
        for veh_id in self.cars_before_ramp:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_RAMP_METER:
                color = self.cars_before_ramp[veh_id]['color']
                self.k.vehicle.set_color(veh_id, color)
                if self.simulator == 'traci':
                    lane_change_mode = self.cars_before_ramp[veh_id][
                        'lane_change_mode']
                    self.k.kernel_api.vehicle.setLaneChangeMode(
                        veh_id, lane_change_mode)
                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            del self.cars_before_ramp[veh_id]

        for lane in range(NUM_RAMP_METERS * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_RAMP_METER][lane]

            for veh_id, pos in cars_in_lane:
                if pos > RAMP_METER_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        if self.simulator == 'traci':
                            # Disable lane changes inside Toll Area
                            lane_change_mode = \
                                self.k.kernel_api.vehicle.getLaneChangeMode(
                                    veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lane_change_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (0, 255, 255))
                        self.cars_before_ramp[veh_id] = {
                            'lane_change_mode': lane_change_mode,
                            'color': color
                        }

    def alinea(self):
        """Utilize the ALINEA algorithm for toll booth metering control.

        This acts as an implementation of the ramp metering control algorithm
        from the article:

        Spiliopoulou, Anastasia D., Ioannis Papamichail, and Markos
        Papageorgiou. "Toll plaza merging traffic control for throughput
        maximization." Journal of Transportation Engineering 136.1 (2009):
        67-76.

        Essentially, we apply feedback control around the value self.n_crit.
        We keep track of the number of vehicles in edge 4, average them across
        time ot get a smoothed value and then compute
        q_{t+1} = clip(q_t + K * (n_crit - n_avg), q_min, q_max). We then
        convert this into a cycle_time value via cycle_time = 7200 / q.
        Cycle_time = self.green_time + red_time i.e. the first self.green_time
        seconds of a cycle will be green, and the remainder will be all red.
        """
        self.feedback_timer += self.sim_step
        self.ramp_state += self.sim_step
        if self.feedback_timer > self.feedback_update_time:
            self.feedback_timer = 0
            # now implement the integral controller update
            # find all the vehicles in an edge
            q_update = self.feedback_coeff * (
                self.n_crit - np.average(self.smoothed_num))
            self.q = np.clip(
                self.q + q_update, a_min=self.q_min, a_max=self.q_max)
            # convert q to cycle time
            self.cycle_time = 7200 / self.q

        # now apply the ramp meter
        self.ramp_state %= self.cycle_time
        # step through, if the value of tl_state is below self.green_time
        # we should be green, otherwise we should be red
        tl_mask = (self.ramp_state <= self.green_time)
        colors = ['G' if val else 'r' for val in tl_mask]
        self.k.traffic_light.set_state('3', ''.join(colors))

    def apply_toll_bridge_control(self):
        """Apply control to the toll bridge.

        Vehicles approaching the toll region slow down and stop lane changing.

        If 'disable_tb' is set to False, vehicles within TOLL_BOOTH_AREA of the
        end of edge EDGE_BEFORE_TOLL are labelled as approaching the toll
        booth. Their color changes and their lane changing is disabled. To
        force them to slow down/mimic the effect of the toll booth, we sample
        from a random normal distribution with mean
        MEAN_NUM_SECONDS_WAIT_AT_TOLL and std-dev 1/self.sim_step to get how
        long a vehicle should wait. We then turn on a red light for that many
        seconds.
        """
        cars_that_have_left = []
        for veh_id in self.cars_waiting_for_toll:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_TOLL:
                lane = self.k.vehicle.get_lane(veh_id)
                color = self.cars_waiting_for_toll[veh_id]["color"]
                self.k.vehicle.set_color(veh_id, color)
                if self.simulator == 'traci':
                    lane_change_mode = \
                        self.cars_waiting_for_toll[veh_id]["lane_change_mode"]
                    self.k.kernel_api.vehicle.setLaneChangeMode(
                        veh_id, lane_change_mode)
                if lane not in self.fast_track_lanes:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                            1 / self.sim_step))
                else:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK /
                            self.sim_step, 1 / self.sim_step))

                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            del self.cars_waiting_for_toll[veh_id]

        traffic_light_states = ["G"] * NUM_TOLL_LANES * self.scaling

        for lane in range(NUM_TOLL_LANES * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_TOLL][lane]

            for veh_id, pos in cars_in_lane:
                if pos > TOLL_BOOTH_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        # Disable lane changes inside Toll Area
                        if self.simulator == 'traci':
                            lane_change_mode = self.k.kernel_api.vehicle.\
                                getLaneChangeMode(veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lane_change_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (255, 0, 255))
                        self.cars_waiting_for_toll[veh_id] = \
                            {'lane_change_mode': lane_change_mode,
                             'color': color}
                    else:
                        if pos > 50:
                            if self.toll_wait_time[lane] < 0:
                                traffic_light_states[lane] = "G"
                            else:
                                traffic_light_states[lane] = "r"
                                self.toll_wait_time[lane] -= 1

        new_tl_state = "".join(traffic_light_states)

        if new_tl_state != self.tl_state:
            self.tl_state = new_tl_state
            self.k.traffic_light.set_state(
                node_id=TB_TL_ID, state=new_tl_state)

    def get_bottleneck_density(self, lanes=None):
        """Return the density of specified lanes.

        If no lanes are specified, this function calculates the
        density of all vehicles on all lanes of the bottleneck edges.
        """
        bottleneck_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        if lanes:
            veh_ids = [
                veh_id for veh_id in bottleneck_ids
                if str(self.k.vehicle.get_edge(veh_id)) + "_" +
                str(self.k.vehicle.get_lane(veh_id)) in lanes
            ]
        else:
            veh_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        return len(veh_ids) / BOTTLE_NECK_LEN

    # Dummy action and observation spaces
    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See parent class.

        To be implemented by child classes.
        """
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        reward = self.k.vehicle.get_outflow_rate(10 * self.sim_step) / \
            (2000.0 * self.scaling)
        return reward

    def get_state(self):
        """See class definition."""
        return np.asarray([1])
