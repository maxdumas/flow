"""Contains a list of custom lane change controllers."""

from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController
import numpy as np

HEADWAY_ALERT = 20

class LaneChangeController_AvoidAVClumping(BaseLaneChangeController):
    """A controller used to enforce sumo lane-change dynamics on a vehicle.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        current_lane = env.k.vehicle.get_lane(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        rl_ids = env.k.vehicle.get_rl_ids()

        h = env.k.vehicle.get_headway(self.veh_id)
        leaders, followers = self.get_lane_leaders_followers(env, rl_ids) 

        # Find the correct direction to change
        if h <= HEADWAY_ALERT and (self.leader_more_than_two_av(env, rl_ids, lead_id) or self.between_two_av(env)):
            
            available_lanes = self.get_available_lanes(env, rl_ids, leaders, followers)

            change_direction = self.get_direction(current_lane, available_lanes)

            return change_direction

        return 0


    def leader_more_than_two_av(self, env, rl_ids, lead_id):
        if lead_id not in rl_ids:
            return False
        else:
            av_lead_id1 = env.k.vehicle.get_leader(lead_id)
            av_lead_id2 = env.k.vehicle.get_leader(av_lead_id1)
            if av_lead_id1 in rl_ids and av_lead_id2 in rl_ids:
                headway1 = env.k.vehicle.get_headway(lead_id)
                headway2 = env.k.vehicle.get_headway(av_lead_id1)
                if headway1 <= HEADWAY_ALERT and headway2 <= HEADWAY_ALERT:
                    print("Car id: {} trigger more than two av".format(self.veh_id))
                    return True
        return False

    def between_two_av(self, env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        follow_id = env.k.vehicle.get_follower(self.veh_id)
        rl_ids = env.k.vehicle.get_rl_ids()
        if lead_id != follow_id and lead_id in rl_ids and follow_id in rl_ids:
            headway = env.k.vehicle.get_headway(follow_id)
            if headway <= HEADWAY_ALERT:
                print("Car id: {} trigger between two av".format(self.veh_id))
                return True
        return False

    def get_available_lanes(self, env, rl_ids, leaders, followers):
        lead_ids = list(leaders.values())
        available_lanes = set()
        for car_id in lead_ids:
            if not self.leader_more_than_two_av(env, rl_ids, car_id):
                available_lanes.add(env.k.vehicle.get_lane(car_id))


        for lane, follower_id in followers.items():
            if lane in leaders:
                headway = env.k.vehicle.get_headway(follower_id)
                if headway >= 2 * HEADWAY_ALERT:
                    available_lanes.add(lane)

        return list(available_lanes)

            

    def get_direction(self, current_lane, available_lanes):
        if not available_lanes:
            return 0

        lane_choices = np.sign(current_lane - np.array(available_lanes))
        if sum(lane_choices) < 0:
            return -1
        elif sum(lane_choices) > 0:
            return 1
        else:
            return np.random.choice([-1, 1])

    def get_lane_leaders_followers(self, env, rl_ids):
        veh_leaders = {}
        veh_followers = {}
        for rl_id in rl_ids:
            followers = env.k.vehicle.get_lane_followers(rl_id)
            leaders = env.k.vehicle.get_lane_leaders(rl_id)
            if self.veh_id in followers:
                veh_leaders[env.k.vehicle.get_lane(rl_id)] = rl_id
            if self.veh_id in leaders:
                veh_followers[env.k.vehicle.get_lane(rl_id)] = rl_id
        return veh_leaders, veh_followers
