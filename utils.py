import gym
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
import os
import torch
from stable_baselines3.common.monitor import Monitor
import glob

def make_env(env_id, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        if log_dir is not None:
            env = Monitor(env,
                          os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)
        return env
    return _thunk


def make_vec_envs(env_name,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    pre_envs = [
        make_env(env_name, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(pre_envs) > 1:
        envs = SubprocVecEnv(pre_envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    return envs

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)



class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info