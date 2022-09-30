"""
Contains several custom car-following control models.

These controllers can be used to modify the acceleration behavior of vehicles
in Flow to match various prominent car-following models that can be calibrated.

Each controller includes the function ``get_accel(self, env) -> acc`` which,
using the current state of the world and existing parameters, uses the control
model to return a vehicle acceleration.
"""
import math
import numpy as np

from flow.controllers.base_controller import BaseController

HEADWAY_ALERT = 20


class IDMController_AvoidAVClumping(BaseController):
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # If there are three AVs in front of the car
        if h <= HEADWAY_ALERT and (self.leader_more_than_two_av(env) or self.between_two_av(env)):
            return v * 0.7


        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)

    def leader_more_than_two_av(self, env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        rl_ids = env.k.vehicle.get_rl_ids()
        if lead_id not in rl_ids:
            return False
        else:
            av_lead_id1 = env.k.vehicle.get_leader(lead_id)
            av_lead_id2 = env.k.vehicle.get_leader(av_lead_id1)
            if av_lead_id1 in rl_ids and av_lead_id2 in rl_ids:
                headway1 = env.k.vehicle.get_headway(lead_id)
                headway2 = env.k.vehicle.get_headway(av_lead_id1)
                if headway1 <= HEADWAY_ALERT and headway2 <= HEADWAY_ALERT:
                    return True
        return False

    def between_two_av(self, env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        follow_id = env.k.vehicle.get_follower(self.veh_id)
        rl_ids = env.k.vehicle.get_rl_ids()
        if lead_id != follow_id and lead_id in rl_ids and follow_id in rl_ids:
            headway = env.k.vehicle.get_headway(follow_id)
            if headway <= HEADWAY_ALERT:
                return True
        return False


