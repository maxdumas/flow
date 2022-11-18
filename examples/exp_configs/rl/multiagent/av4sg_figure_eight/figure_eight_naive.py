"""Figure eight example."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import IDMController, ContinuousRouter, RLController, \
                             IDMController_AvoidAVClumping, LaneChangeController_AvoidAVClumping
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env
from flow.envs import MultiAgentFigureEightNaiveEnv

# time horizon of a single rollout
HORIZON = 5000
# number of rollouts per training iteration
# N_ROLLOUTS = 20
N_ROLLOUTS = 5
N_STEPS = 5
# number of parallel workers
# N_CPUS = 2
N_CPUS = 3
N_GPUS = 1
# desired velocity for all vehicles in the network, in m/s
TARGET_VELOCITY = 30
# maximum acceleration for autonomous vehicles, in m/s^2
MAX_ACCEL = 1
# maximum deceleration for autonomous vehicles, in m/s^2
MAX_DECEL = 1

# number of automated vehicles. Must be one of [1, 2, 7, 14]
NUM_AUTOMATED = 5


additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # radius of the circular components
    "radius_ring": 50,
    # number of lanes
    "lanes": 3,
    # speed limit for all edges
    "speed_limit": 40,
    # resolution of the curved portions
    "resolution": 40
})

# We evenly distribute the autonomous vehicles in between the human-driven
# vehicles in the network.
num_human = 20 - NUM_AUTOMATED
human_per_automated = int(num_human / NUM_AUTOMATED)

vehicles = VehicleParams()
for i in range(NUM_AUTOMATED):
    vehicles.add(
        veh_id='human_{}'.format(i),
        acceleration_controller=(IDMController_AvoidAVClumping, {
            "noise": 0.2
        }),
        lane_change_controller=(LaneChangeController_AvoidAVClumping, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            decel=1.5,
        ),
        lane_change_params=SumoLaneChangeParams(lane_change_mode=1621),
        num_vehicles=human_per_automated)
    # vehicles.add(
    #     veh_id='human_{}'.format(i),
    #     acceleration_controller=(IDMController, {
    #         'noise': 0.2
    #     }),
    #     routing_controller=(ContinuousRouter, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         speed_mode="obey_safe_speed",
    #         decel=1.5,
    #     ),
    #     num_vehicles=human_per_automated)
    vehicles.add(
        veh_id='rl_{}'.format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            accel=MAX_ACCEL,
            decel=MAX_DECEL,
        ),
        num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag='figure_eight_naive',

    # name of the flow environment the experiment is running on
    env_name=MultiAgentFigureEightNaiveEnv,

    # name of the network class the experiment is running on
    network=FigureEightNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        overtake_right=True,
        restart_instance=True,
        print_warnings=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': TARGET_VELOCITY,
            'max_accel': MAX_ACCEL,
            'max_decel': MAX_DECEL,
            'sort_vehicles': False,
            "lane_change_duration": 3
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


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
