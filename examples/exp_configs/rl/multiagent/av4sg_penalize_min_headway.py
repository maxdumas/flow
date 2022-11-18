"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from flow.controllers import RLController
from flow.core.params import EnvParams, NetParams, InitialConfig, InFlows, \
                             VehicleParams, SumoParams, \
                             SumoCarFollowingParams, SumoLaneChangeParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks import HighwayRampsNetwork, HighwayNetwork
from flow.envs import MultiAgentHighwayFancyEnv
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env


# SET UP PARAMETERS FOR THE SIMULATION

# number of training iterations
N_TRAINING_ITERATIONS = 200
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of steps per rollout
HORIZON = 1500
# number of parallel workers
N_CPUS = 7

# inflow rate on the highway in vehicles per hour
HIGHWAY_INFLOW_RATE = 4000
# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 20


# SET UP PARAMETERS FOR THE NETWORK

additional_net_params = {
    # length of the highway
    "length": 1500,
    # number of lanes
    "lanes": 3,
    # speed limit for all edges
    "speed_limit": 40,
    # number of edges to divide the highway into
    "num_edges": 1,
    # whether to include a ghost edge. This edge is provided a different speed
    # limit.
    "use_ghost_edge": False,
    # speed limit for the ghost edge
    "ghost_speed_limit": 25,
    # length of the downstream ghost edge with the reduced speed limit
    "boundary_cell_length": 500
}


# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 1,
    'max_decel': 1,
    'target_velocity': 30,
    'lane_change_duration': 3
})


# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()

# human vehicles
vehicles.add(
    veh_id="idm",
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",  # for safer behavior at the merges
        tau=1.5  # larger distance between cars
    ),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621))

# autonomous vehicles
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}))

# add human vehicles on the highway
inflows.add(
    veh_type="idm",
    edge="highway_0",
    vehs_per_hour=HIGHWAY_INFLOW_RATE,
    depart_lane="free",
    depart_speed="max",
    name="idm_highway_inflow")

# add autonomous vehicles on the highway
# they will stay on the highway, i.e. they won't exit through the off-ramps
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * PENETRATION_RATE / 100),
    depart_lane="free",
    depart_speed="max",
    name="rl_highway_inflow",
    route="routehighway_0_0")


# SET UP FLOW PARAMETERS

flow_params = dict(
    # name of the experiment
    exp_tag='multiagent_highway',

    # name of the flow environment the experiment is running on
    env_name=MultiAgentHighwayFancyEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=200,
        sims_per_step=1,  # do not put more than one
        additional_params=additional_env_params,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True,
        overtake_right=True,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflows,
        additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


# SET UP RLLIB MULTI-AGENT FEATURES

create_env, env_name = make_create_env(params=flow_params, version=0)

# register as rllib env
register_env(env_name, create_env)

# multiagent configuration
test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


POLICY_GRAPHS = {'av': (PPOTFPolicy, obs_space, act_space, {})}

POLICIES_TO_TRAIN = ['av']


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'
