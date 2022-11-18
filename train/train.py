"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""
import argparse
import json
import os
import sys
from time import strftime
from copy import deepcopy

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="rllib",
        help='the RL trainer to use. either rllib or Stable-Baselines')

    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]

def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_gpus,
                     n_rollouts,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from ray import tune
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    horizon = flow_params['env'].horizon

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    # Hyper-parameter doc: https://docs.ray.io/en/latest/rllib/rllib-training.html
    config["num_workers"] = n_cpus
    config["num_gpus"] = n_gpus
    config["train_batch_size"] = horizon * n_rollouts
    config["gamma"] = 0.995  # discount rate
    config["model"].update({"fcnet_hiddens": [64, 64]})
    config["model"].update({"fcnet_activation": "tanh"})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon
    config["framework"] = 'torch'
    config['batch_mode'] = 'complete_episodes'
    config['rollout_fragment_length'] = 500
    # config['multiagent'].update({'count_steps_by': 'agent_steps'})

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        print("policy_graphs", policy_graphs)
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config

def gen_exp(submodule, flags):
    """Generate experiments from exp_config."""
    import ray

    flow_params = submodule.flow_params
    n_cpus = submodule.N_CPUS
    n_gpus = submodule.N_GPUS
    n_rollouts = submodule.N_ROLLOUTS
    n_steps = submodule.N_STEPS
    policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, n_cpus, n_gpus, n_rollouts,
        policy_graphs, policy_mapping_fn, policies_to_train)

    # ray.init(num_cpus=n_cpus + 1, num_gpus=n_gpus, object_store_memory=200 * 1024 * 1024)
    exp_config = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 1,
        "checkpoint_at_end": True,
        "max_failures": 1,
        "stop": {
            "training_iteration": n_steps,
        },
    }
    return flow_params["exp_tag"], exp_config

def train_rllib(experiments, flags, concurrent=False, reuse_actors=True):
    """Return the relevant components of an RLlib experiment.

        Parameters
        ----------
        experiments : dict
            {tag1: exp_config1, tag2: exp_config2, ...}

        flags : flags

        concurrent : bool
            Whether running experiments in parallel or one by one

        reuse_actors : bool
            If your trainable is slow to initialize, consider setting reuse_actors=True
            to reduce actor creation overheads

        Returns
        -------
        None
        """
    import ray
    from ray.tune import run_experiments

    ray.init(num_cpus= 12, num_gpus=1, object_store_memory=1000 * 1024 * 1024)

    if len(experiments) == 1 and flags.checkpoint_path is not None:
        exp_config['restore'] = flags.checkpoint_path
    # experiments
    run_experiments(experiments, concurrent=concurrent, reuse_actors=reuse_actors)

def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)
    # store the experiments as dict and pass it in run_experiments
    experiments = {}
    exps_to_train = flags.exp_config
    with open(exps_to_train) as f:
        lines = f.readlines()
    for line in lines:
        exp_config = line.strip()

        # Import relevant information from the exp_config script.
        module = __import__(
            "examples.exp_configs.rl.singleagent", fromlist=[exp_config])
        module_ma = __import__(
            "examples.exp_configs.rl.multiagent", fromlist=[exp_config])

        # Import the sub-module containing the specified exp_config and determine
        # whether the environment is single agent or multi-agent.
        if hasattr(module, exp_config):
            submodule = getattr(exp_config)
            multiagent = False
        elif hasattr(module_ma, exp_config):
            submodule = getattr(module_ma, exp_config)
            assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
                "Currently, multiagent experiments are only supported through "\
                "RLlib. Try running this experiment using RLlib: " \
                "'python train.py EXP_CONFIG'"
            multiagent = True
        else:
            raise ValueError("Unable to find experiment config.")

        # Perform the training operation.
        if flags.rl_trainer.lower() == "rllib":
           tag, config = gen_exp(submodule, flags)
           experiments[tag] = config
        else:
            raise ValueError("rl_trainer should be either 'rllib', 'h-baselines', "
                             "or 'stable-baselines'.")
    train_rllib(experiments, flags)

if __name__ == "__main__":
    main(sys.argv[1:])

# def train_rllib(submodule, flags):
#     """Train policies using the PPO algorithm in RLlib."""
#     import ray
#     from ray.tune import run_experiments
#
#     flow_params = submodule.flow_params
#     n_cpus = submodule.N_CPUS
#     n_gpus = submodule.N_GPUS
#     n_rollouts = submodule.N_ROLLOUTS
#     policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
#     policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
#     policies_to_train = getattr(submodule, "policies_to_train", None)
#
#     alg_run, gym_name, config = setup_exps_rllib(
#         flow_params, n_cpus, n_gpus, n_rollouts,
#         policy_graphs, policy_mapping_fn, policies_to_train)
#
#     ray.init(num_cpus=n_cpus + 1, num_gpus=n_gpus, object_store_memory=200 * 1024 * 1024)
#     exp_config = {
#         "run": alg_run,
#         "env": gym_name,
#         "config": {
#             **config
#         },
#         "checkpoint_freq": 1,
#         "checkpoint_at_end": True,
#         "max_failures": 1,
#         "stop": {
#             "training_iteration": flags.num_steps,
#         },
#     }
#
#     if flags.checkpoint_path is not None:
#         exp_config['restore'] = flags.checkpoint_path
#     run_experiments({flow_params["exp_tag"]: exp_config})

# def main(args):
#     """Perform the training operations."""
#     # Parse script-level arguments (not including package arguments).
#     flags = parse_args(args)
#
#     # Import relevant information from the exp_config script.
#     module = __import__(
#         "examples.exp_configs.rl.singleagent", fromlist=[flags.exp_config])
#     module_ma = __import__(
#         "examples.exp_configs.rl.multiagent", fromlist=[flags.exp_config])
#
#     # Import the sub-module containing the specified exp_config and determine
#     # whether the environment is single agent or multi-agent.
#     if hasattr(module, flags.exp_config):
#         submodule = getattr(module, flags.exp_config)
#         multiagent = False
#     elif hasattr(module_ma, flags.exp_config):
#         submodule = getattr(module_ma, flags.exp_config)
#         assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
#             "Currently, multiagent experiments are only supported through "\
#             "RLlib. Try running this experiment using RLlib: " \
#             "'python train.py EXP_CONFIG'"
#         multiagent = True
#     else:
#         raise ValueError("Unable to find experiment config.")
#
#     # Perform the training operation.
#     if flags.rl_trainer.lower() == "rllib":
#         train_rllib(submodule, flags)
#     else:
#         raise ValueError("rl_trainer should be either 'rllib', 'h-baselines', "
#                          "or 'stable-baselines'.")


