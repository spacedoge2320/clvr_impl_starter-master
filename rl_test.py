import gym
import sprites_env
import cv2
import time
import cnn_baseline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import os
import numpy as np
from collections import deque
import utils
from evaluation import evaluate
from model import Policy
from storage import RolloutStorage
import algorithms.PPO_algo as PPO_algo


# Check if the MPS (Metal Performance Shaders) backend is available
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("MPS is available")
else:
    device = torch.device('cpu')
    print("MPS is not available")


def main():
    num_env_steps = 1000000
    num_steps = 128
    num_processes = 8
    num_mini_batch = 4
    clip_param = 0.1
    lr = 2.5e-4
    use_linear_lr_decay = True
    entropy_coef = 0.01
    env_name = 'Sprites-v1'
    seed = 1242
    log_dir = 'logs'
    save_dir = 'model_weights'
    save_interval = 100
    log_interval = 1
    algo = 'PPO'
    use_gae = False
    gae_lambda = 0.95
    gamma = 0.99
    use_proper_time_limits = False
    eval_interval = 10000
    recurrent_policy = False

    log_dir = os.path.expanduser(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.manual_seed(seed)

    envs = utils.make_vec_envs(env_name, seed, num_processes,
                    gamma, log_dir, device, False)

    actor_critic = Policy(
        (envs.observation_space.shape),
        envs.action_space,
        base_kwargs={'recurrent': recurrent_policy})
    actor_critic.to(device)
    
    if algo == 'PPO':
        agent = PPO_algo.PPO(
            actor_critic=actor_critic,
            clip_param=clip_param,
            ppo_epoch=4,
            num_mini_batch=num_mini_batch,
            entropy_coef=entropy_coef,
            value_loss_coef=0.5,
            lr=lr,
            max_grad_norm=0.5
            )

    rollouts = RolloutStorage(num_steps, num_processes,
                            envs.observation_space.shape, envs.action_space,
                            actor_critic.recurrent_hidden_state_size)
        
    # All RL baselines should have following interface:
    # RL_policy.act(observation) -> action
    # RL_policy.update(observation, reward, done, info) -> Loss?
    episode_rewards = deque(maxlen=10)

    obs = envs.reset()

    start = time.time()
    num_updates = int(
        num_env_steps) // num_steps // num_processes
    for j in range(num_updates):

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device),
                    rollouts.masks[step].to(device))

        obs, reward, done, infos = envs.step(action)  # 액션 수행 및 결과 받기

        for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

        masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action,
            action_log_prob, value, reward, masks, bad_masks)
        
        
        #print(obs.shape)
        
    with torch.no_grad():
        next_value = actor_critic.get_value(
        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
        rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, use_gae, gamma,
                                 gae_lambda, use_proper_time_limits)
    
    value_loss, action_loss, dist_entropy = agent.update(rollouts)
    rollouts.after_update()

    # save for every interval-th episode or for the last epoch
    if (j % save_interval == 0
            or j == num_updates - 1) and save_dir != "":
        save_path = os.path.join(save_dir, algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        torch.save([
            actor_critic,
            getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
        ], os.path.join(save_path, env_name + ".pt"))

    if j % log_interval == 0 and len(episode_rewards) > 1:
        total_num_steps = (j + 1) * num_processes * num_steps
        end = time.time()
        print(
            "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            .format(j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss,
                    action_loss))
        
        obs_copy = obs.cpu().clone().detach().numpy()
        image = None
        for i in range (obs_copy.shape[0]):
            observation_resized = cv2.resize(obs_copy[0,0,:,:], (500, 500))
            if image is None:
                image = observation_resized
            else:
                image = np.hstack((image, observation_resized))
        cv2.imshow('Environment', image)
        cv2.waitKey(1)

    if (eval_interval is not None and len(episode_rewards) > 1
            and j % eval_interval == 0):
        obs_rms = utils.get_vec_normalize(envs).obs_rms
        evaluate(actor_critic, obs_rms, env_name, seed,
                num_processes, eval_log_dir, device)


if __name__ == '__main__':
    main()
