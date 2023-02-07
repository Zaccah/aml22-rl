"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, randomize = False):
        MujocoEnv.__init__(self, 4, randomize)
        utils.EzPickle.__init__(self)


        self.bounds = None
        self.done = False
        self.domain = domain

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0

        self.mean_rand = self.model.body_mass[2:]
        self.mean_max = [val * 1.75 for val in self.mean_rand]
        self.mean_min = [val * 0.25 for val in self.mean_rand]


    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(*self.sample_parameters())

    def set_bounds(self, bounds: list):
        self.bounds = bounds

    def set_std(self, stand_dev):
        self.stand_dev = [value * stand_dev for value in self.mean_rand]

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        if self.bounds is not None:
            rand_masses = [np.random.uniform(l, h) for (l, h) in self.bounds]
            rand_masses.insert(0, self.sim.model.body_mass[1])
            return rand_masses
        
        rand_masses = []
        for i in range(len(self.stand_dev)):
            rand_masses.append(np.random.normal(self.mean_rand[i], self.stand_dev[i]))
            if rand_masses[i] > self.mean_max[i]:
                rand_masses[i] = self.mean_max[i]
            if rand_masses[i] < self.mean_min[i]:
                rand_masses[i] = self.mean_min[i]
        rand_masses.insert(0, self.sim.model.body_mass[1])
        return rand_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, *task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        self.done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, self.done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="CustomHopper-source-rand-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "randomize": True}
)

gym.envs.register(
        id="CustomHopper-target-rand-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target", "randomize": True}
)

