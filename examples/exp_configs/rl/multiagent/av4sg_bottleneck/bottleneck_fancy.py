import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import IDMController, ContinuousRouter, RLController, \
                             IDMController_AvoidAVClumping, LaneChangeController_AvoidAVClumping, \
                             SimLaneChangeController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import TrafficLightParams
from flow.networks.bottleneck import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.networks import BottleneckNetwork
from flow.utils.registry import make_create_env
from flow.envs import MultiAgentBottleneckDesiredThroughputEnv_Fancy

# training configuration
# time horizon of a single rollout
HORIZON = 1000
# number of parallel workers
N_CPUS = 5
N_GPUS = 1
# number of rollouts per training iteration
# N_ROLLOUTS = N_CPUS * 4
N_ROLLOUTS = 20
N_STEPS = 50

SCALING = 1
NUM_LANES = 4 # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.25

# add vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController_AvoidAVClumping, {
        "noise": 0.2
    }),
    lane_change_controller=(LaneChangeController_AvoidAVClumping, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),
    num_vehicles=1 * SCALING)
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),
    num_vehicles=1 * SCALING)

# bottleneck network configuration
controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
additional_env_params = {
    "target_velocity": 20,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "controlled_segments": controlled_segments,
    "symmetric": False,
    "observed_segments": num_observed_segments,
    "reset_inflow": False,
    "lane_change_duration": 3,
    "max_accel": 1,
    "max_decel": 1,
    "inflow_range": [2500, 3500]
}

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    "scaling": SCALING,
    "speed_limit": 30
})

flow_rate = 3000
# percentage of flow coming out of each lane
inflow = InFlows()
inflow.add(
    veh_type="idm",
    edge="1",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    name="idm",
    depart_lane="random",
    depart_speed=10)
inflow.add(
    veh_type="rl",
    edge="1",
    vehs_per_hour=flow_rate * AV_FRAC,
    name="rl",
    depart_lane="random",
    depart_speed=10)

# configuration of traffic lights
# currently no traffic lights
traffic_lights = TrafficLightParams()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

flow_params = dict(
    # name of the experiment
    exp_tag="bottleneck_fancy",

    # name of the flow environment the experiment is running on
    env_name=MultiAgentBottleneckDesiredThroughputEnv_Fancy,
    # name of the network class the experiment is running on
    network=BottleneckNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        print_warnings=False,
        restart_instance=True,
        overtake_right=True,
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
        min_gap=2.5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"],
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    # tls=traffic_lights,
)


# configuration for multi-agents
create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTorchPolicy, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']