"""This module provides functions that are intended to provide modular fragments
of state that go well together."""

from functools import partial

import numpy as np
from numpy.typing import NDArray

from flow.envs.base import Env

def get_surrounding_headways(env: Env, veh_id: str) -> NDArray[np.float32]:
    r"""
    Given a vehicle ID, returns a 6-length array containing the headways of the
    surrounding cars. Headways are normalized relative the entire length of the
    network. A value of 0.0 means the vehicles are in the same position, and a
    value of 1.0 means the vehicles are as far from one another as the network
    allows.
    
    See the below diagram for an explanation of the order and significance of
    each array element:

    0  1  2
     \ | /
    ---O---
     / | \
    3  4  5

    In the above diagram, O is the vehicle indicated by "veh_id", and each
    number is the index of the resulting headway in the returned array. Headways
    are measured as distance along the lane of the measured vehicle from the
    middle horizontal line.

    If no vehicle exists in front or behind the current vehicle in the current
    lane, then 1.0 is returned in the corresponding headway array entry,
    representing the maximum possible distance in the network.
    """
    max_length = env.k.network.length()
    num_lanes = env.k.network.num_lanes(env.k.vehicle.get_edge(veh_id))

    current_lane: int = env.k.vehicle.get_lane(veh_id)  # type: ignore
    left_neighbor_lane = current_lane - 1
    right_neighbor_lane = current_lane + 1

    def get_headway(l_ids: list[str], headways: list[str], l: int):
        if l < 0 or l > num_lanes - 1:
            return 1.0
        return headways[l] if l_ids[l] not in {"", None} else 1.0

    followers = env.k.vehicle.get_lane_followers(veh_id)
    follower_headways = env.k.vehicle.get_lane_tailways(veh_id)

    leaders = env.k.vehicle.get_lane_leaders(veh_id)
    leader_headways = env.k.vehicle.get_lane_tailways(veh_id)

    get_follower_headway = partial(get_headway, followers, follower_headways)
    get_leader_headway = partial(get_headway, leaders, leader_headways)

    ls = [left_neighbor_lane, current_lane, right_neighbor_lane]

    return (
        np.array(
            [get_leader_headway(l) for l in ls] + [get_follower_headway(l) for l in ls]
        )
        / max_length
    )
