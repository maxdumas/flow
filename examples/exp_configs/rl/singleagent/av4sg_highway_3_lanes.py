"""Example of an open multi-lane network with human-driven vehicles."""
"""Bottleneck example.

Bottleneck in which the actions are specifying a desired velocity
in a segment of space
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController, IDMController_AvoidAVClumping, LaneChangeController_AvoidAVClumping
from flow.envs import BottleneckDesiredVelocityEnv
from flow.networks import BottleneckNetwork
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv
from flow.envs import HighwayRegulationEnv

# time horizon of a single rollout
HORIZON = 2000
# number of parallel workers
N_CPUS = 5
# number of rollouts per training iteration
# N_ROLLOUTS = N_CPUS * 4
N_ROLLOUTS = 5


SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway

AV_FRAC = 0.25

vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController_AvoidAVClumping, {
        "noise": 0.2
    }),
    lane_change_controller=(LaneChangeController_AvoidAVClumping, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=0
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),
    routing_controller=(ContinuousRouter, {}))

# autonomous vehicles
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),)

additional_env_params = {
    "target_velocity": 40,
    "reset_inflow": False,
    "lane_change_duration": 5,
    "max_accel": 3,
    "max_decel": 3,
    "inflow_range": [1000, 2000],
    "sort_vehicles": False,
    "observed_segments": [("highway_0", 5)]
}

# flow rate
flow_rate = 2300 * SCALING

# percentage of flow coming out of each lane
inflow = InFlows()
inflow.add(
    veh_type="idm",
    edge="highway_0",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    depart_lane="random",
    depart_speed=10)
inflow.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=flow_rate * AV_FRAC,
    depart_lane="random",
    depart_speed=10)


additional_net_params = {
    "scaling": SCALING, 
    "speed_limit": 23,
    # length of the highway
    "length": 1000,
    # number of lanes
    "lanes": 4,
    # number of edges to divide the highway into
    "num_edges": 1,
    "use_ghost_edge": False,
    # speed limit for the ghost edge
    "ghost_speed_limit": 25,
    # length of the downstream ghost edge with the reduced speed limit
    "boundary_cell_length": 500,
    "scaling": SCALING
}

net_params = NetParams(
    inflows=inflow,
    additional_params=additional_net_params)

flow_params = dict(
    # name of the experiment
    exp_tag="highway",

    # name of the flow environment the experiment is running on
    env_name=HighwayRegulationEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps=40,
        sims_per_step=1,
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf")
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    # tls=traffic_lights,
)