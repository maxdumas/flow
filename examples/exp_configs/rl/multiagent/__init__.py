"""Contains all callable environments in Flow."""
from examples.exp_configs.rl.multiagent.av4sg_figure_eight import figure_eight_naive, figure_eight_fancy
from examples.exp_configs.rl.multiagent.av4sg_highway_ramps import on_off_ramps_fancy

__all__ = [
    'figure_eight_naive',
    'figure_eight_fancy',
    'on_off_ramps_fancy'
]
