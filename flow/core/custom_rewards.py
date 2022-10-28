"""A series of reward functions."""

import numpy as np
from scipy.stats import norm
from scipy.special import expit


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


def fancy_reward(env, dists_to_leader, dists_to_follower, min_dist_to_leader, min_dist_to_follower, fail=False):

    # Reward component that penalizes velocities that are close to 0
    def r_moving_function(v):
        k = 10 # How quickly the reward will decrease to 0 once we are approaching 0 velocity.
        v_0 = 1 # The speed below which we consider ourselves moving too slowly. "Stopped".
        return expit(k * (v - v_0))

    # Reward component to penalize headways that are too small
    def r_penalize_too_close_function(dist, dist_min):
        k = 5
        return expit(k * (dist - dist_min))

    # Reward component that rewards velocities that are cose to the target velocity and nothing else
    def r_target_function(v, v_t, spread):
        D = norm(loc=v_t, scale=spread)
        return D.pdf(v)

    target_vel = env.env_params.additional_params["target_velocity"]
    r_target_spread = 0.8
    veh_ids = env.k.vehicle.get_ids()
    vels = np.array(env.k.vehicle.get_speed(veh_ids))

    # Calculate r_moving from the lowest velocity. If any vehicle is stopped, we
    # want the reward to be 0.
    r_is_moving = r_moving_function(vels.min())

    # For r_target_vel, we calculate the reward across all velocities and take the
    # sum, then normalize it against the maximum possible r_target_vel score, which
    # would be 1 if every vehicle was traveling at exactly the target speed
    r_target_vel = r_target_function(vels, target_vel, r_target_spread).sum()
    max_r_target = r_target_function(target_vel, target_vel, r_target_spread) * len(vels)
    r_target_vel /= max_r_target

    # no leader
    if dists_to_leader == -1:
        r_penalize_too_close_to_leader = 1
    else:
        r_penalize_too_close_to_leader = r_penalize_too_close_function(dists_to_leader, min_dist_to_leader).mean()

    if dists_to_follower == -1:
        r_penalize_too_close_to_follower = 1
    else:
        r_penalize_too_close_to_follower = r_penalize_too_close_function(dists_to_follower, min_dist_to_follower).mean()


    return 10.0 * r_is_moving * r_penalize_too_close_to_leader * r_penalize_too_close_to_follower * r_target_vel