"""A series of reward functions."""
import numpy as np
from scipy.stats import norm
from scipy.special import expit
from flow.core.state_fragments import get_surrounding_headways

from flow.envs.base import Env


def dumas_reward(env, fail=False, edge_list=None):
    if edge_list is None:
        veh_ids = env.k.vehicle.get_ids()
    else:
        veh_ids = env.k.vehicle.get_ids_by_edge(edge_list)

    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.0

    target_vel = env.env_params.additional_params["target_velocity"]

    s = 0.7
    l = 10
    k = 5
    v_0 = 1

    def r_moving_function(v):
        return 1 / (1 + np.exp(-k * (v - v_0)))

    def r_target_function(v):
        return l * np.exp(-0.5 * (v - target_vel) ** 2) / (s * np.sqrt(2 * np.pi))

    calculate_r_target = np.vectorize(r_target_function)
    r_moving = r_moving_function(vel.min())
    r_target = calculate_r_target(vel).sum()
    max_r_target = r_target_function(target_vel)

    return r_target * r_moving / max_r_target


def exp(x):
    """
    This is a modified sigmoid function that will return ~0 at x <= 0, and ~1
    at x >= 1, with an S-shaped curve for the values in between.
    """
    return expit(6 * (2 * x - 1))


def penalize_close_to_stopping(v: float) -> float:
    """
    Reward component that penalizes velocities that are close to 0
    """
    k = 10  # How quickly the reward will decrease to 0 once we are approaching 0 velocity.
    v_0 = 1  # The speed below which we consider ourselves moving too slowly. "Stopped".
    return expit(k * (v - v_0))


def penalize_too_close_to_others(dist: float, dist_min: float) -> float:
    """
    Reward component to penalize headways that are too small.
    """
    if dist < 0:
        # Assume the dist is invalid and ignore this reward, if < 0
        return 1.0
    return exp(dist / dist_min)


def reward_target_velocity(v: float, v_t: float, spread: float):
    """
    Reward component that rewards velocities that are close to the target
    velocity and nothing else.
    """
    D = norm(loc=v_t, scale=spread)
    return D.pdf(v)


def reward_high_efficiency(env: Env, time_span=10):
    """Reward component that returns the ratio of inflows to outflows over
    time_span. Note that this reward only makes sense in open networks."""
    inflow = env.k.vehicle.get_inflow_rate(10 * env.sim_step)
    inflow = inflow if inflow != 0.0 else 2000.0
    return env.k.vehicle.get_outflow_rate(time_span * env.sim_step) / inflow


def fancy_reward(
    env,
    dists_to_leader,
    dists_to_follower,
    min_dist_to_leader,
    min_dist_to_follower,
    fail=False,
):
    target_vel = env.env_params.additional_params["target_velocity"]
    r_target_spread = 3.0
    veh_ids = env.k.vehicle.get_ids()
    vels = np.array(env.k.vehicle.get_speed(veh_ids))

    # Calculate r_moving from the lowest velocity. If any vehicle is stopped, we
    # want the reward to be 0.
    r_is_moving = penalize_close_to_stopping(vels.min())

    # For r_target_vel, we calculate the reward across all velocities and take the
    # sum, then normalize it against the maximum possible r_target_vel score, which
    # would be 1 if every vehicle was traveling at exactly the target speed
    r_target_vel = reward_target_velocity(vels, target_vel, r_target_spread).sum()
    max_r_target = reward_target_velocity(
        target_vel, target_vel, r_target_spread
    ) * len(vels)
    r_target_vel /= max_r_target

    # no leader
    if dists_to_leader == -1:
        r_penalize_too_close_to_leader = 1
    else:
        r_penalize_too_close_to_leader = penalize_too_close_to_others(
            dists_to_leader, min_dist_to_leader
        ).mean()

    if dists_to_follower == -1:
        r_penalize_too_close_to_follower = 1
    else:
        r_penalize_too_close_to_follower = penalize_too_close_to_others(
            dists_to_follower, min_dist_to_follower
        ).mean()

    return (
        10.0
        * r_is_moving
        * r_penalize_too_close_to_leader
        * r_penalize_too_close_to_follower
        * r_target_vel
    )


def naive_reward(
    env,
    dists_to_leader,
    dists_to_follower,
    min_dist_to_leader,
    min_dist_to_follower,
    fail=False,
):
    target_vel = env.env_params.additional_params["target_velocity"]
    r_target_spread = 3.0
    veh_ids = env.k.vehicle.get_ids()
    vels = np.array(env.k.vehicle.get_speed(veh_ids))

    # Calculate r_moving from the lowest velocity. If any vehicle is stopped, we
    # want the reward to be 0.
    r_is_moving = penalize_close_to_stopping(vels.min())

    # For r_target_vel, we calculate the reward across all velocities and take the
    # sum, then normalize it against the maximum possible r_target_vel score, which
    # would be 1 if every vehicle was traveling at exactly the target speed
    r_target_vel = reward_target_velocity(vels, target_vel, r_target_spread).sum()
    max_r_target = reward_target_velocity(
        target_vel, target_vel, r_target_spread
    ) * len(vels)
    r_target_vel /= max_r_target

    return 10.0 * r_is_moving * r_target_vel


def naive_reward_2(env: Env, veh_id: str):
    """A reward function which rewards a vehicle that is not stopped and in a
    system with high throughput efficiency."""
    # If the vehicle indicated by veh_id is almost stopped, we want the reward to be 0
    r_is_moving = penalize_close_to_stopping(env.k.vehicle.get_speed(veh_id))
    r_throughput_efficiency = reward_high_efficiency(env)

    return r_is_moving * r_throughput_efficiency


def fancy_reward_2(
    env: Env,
    veh_id: str,
):
    """
    A reward function which rewards a vehicle that is not stopped and in a
    system with high throughput efficiency, and penalizes the vehicle if it is
    too close to its leader or follower.

    :param desired_dist_to_leader The distance that the vehicle should maintain to
    the vehicle in front of it when traveling at the speed limit. Note that this
    value will be smoothly interpolated to zero as the vehicle's speed decreases
    below the speed limit.

    :param desired_dist_to_follower The distance that the vehicle should maintain to
    the vehicle behind it when traveling at the speed limit. Note that this
    value will be smoothly interpolated to zero as the vehicle's speed decreases
    below the speed limit.
    """
    max_length = env.k.network.length()

    r_naive = naive_reward_2(env, veh_id)

    v = env.k.vehicle.get_speed(veh_id)
    speed_limit = env.net_params.additional_params["speed_limit"]

    # An s-shaped curve from 0 at v=0 to 1 at v=speed_limit
    s = exp(v / speed_limit)

    lead_id = env.k.vehicle.get_leader(veh_id)
    desired_dist_to_leader = env.k.vehicle.get_distance_preference(lead_id) / max_length

    follower_id = env.k.vehicle.get_follower(veh_id)
    desired_dist_to_follower = (
        env.k.vehicle.get_distance_preference(follower_id) / max_length
    )

    headways_normalized = get_surrounding_headways(env, veh_id)

    r_penalize_too_close_to_leader = penalize_too_close_to_others(
        headways_normalized[1], s * desired_dist_to_leader
    )

    r_penalize_too_close_to_follower = penalize_too_close_to_others(
        headways_normalized[4], s * desired_dist_to_follower
    )

    return r_penalize_too_close_to_leader * r_penalize_too_close_to_follower * r_naive
