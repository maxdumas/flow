"""Figure eight example."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs import AccelEnv
from flow.networks import FigureEightNetwork

# time horizon of a single rollout
HORIZON = 5000
# number of rollouts per training iteration
N_ROLLOUTS = 2
# number of parallel workers
N_CPUS = 10

# Number of vehicles. We want to test 10% AV penetration and 20% AV penetration
# 25% penetration
N_VEHICLES = 20

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()

for i in range(N_VEHICLES):
    if i in {0, 5}:
        vehicles.add(
            veh_id=f"rl_{i}",
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
                decel=1.5,
            ),
            num_vehicles=1,
        )
    else:
        vehicles.add(
            veh_id=f"human_{i}",
            acceleration_controller=(IDMController, {"noise": 0.2}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
                decel=1.5,
            ),
            num_vehicles=1,
        )


flow_params = dict(
    # name of the experiment
    exp_tag="10p_av4sg_singleagent_figure_eight",
    # name of the flow environment the experiment is running on
    env_name=AccelEnv,
    # name of the network class the experiment is running on
    network=FigureEightNetwork,
    # simulator that is used by the experiment
    simulator="traci",
    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),
    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 20,
            "max_accel": 3,
            "max_decel": 3,
            "sort_vehicles": False,
        },
    ),
    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),
    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,
    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
