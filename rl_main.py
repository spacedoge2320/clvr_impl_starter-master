import copy
import glob
import os
import time
from collections import deque
import sprites_env
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import yaml
warnings.filterwarnings("ignore", category=DeprecationWarning)
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from tqdm import tqdm
import csv
import datetime
from google.cloud import storage
import yaml

class RL_main:
    def __init__(self, config, gcp_bucket=None):
        Environment_list = ["Sprites", "SpritesState"]
        architecture = config['architecture_list'][config['architecture']]

        if architecture == "Oracle":
            env = Environment_list[1] + "-v"+str(config['distractors'])
            config['pretrained_encoder'] = False
        if architecture == "CNN":
            env = Environment_list[0] + "-v"+str(config['distractors'])
            config['pretrained_encoder'] = False
        if architecture == "Pre-trained":
            env = Environment_list[0] + "-v"+str(config['distractors'])
            config['pretrained_encoder'] = True

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        save_name = architecture + "_v" + str(config['distractors']) + "_"+str(current_time)+"_"+config['log_comment']
        config['save_name'] = save_name
        config['env_name'] = env

        self.main(config)

    def load_model_weights(self, actor_critic, envs, save_path, device):
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            actor_critic.load_state_dict(checkpoint[0].state_dict())
            vec_norm = utils.get_vec_normalize(envs)
            if vec_norm is not None and checkpoint[1] is not None:
                vec_norm.obs_rms = checkpoint[1]
            print(f"Loaded model weights from {save_path}")
        else:
            print(f"No model weights found at {save_path}")

    def load_encoder_weights(self, actor_critic, save_path, device):
        checkpoint = torch.load(save_path, map_location=device)
        if 'encoder_state_dict' in checkpoint:
            actor_critic.feature_extractor.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"Loaded encoder weights from {save_path}")
            print(actor_critic.feature_extractor.state_dict().keys())
        else:
            raise KeyError("Pretrained encoder weights not found in checkpoint")

    def main(self, config):
        if config['cuda'] + config['cpu'] + config['mps'] !=1:
            raise ValueError("Only one of CUDA, CPU, or MPS can be enabled at a time")

        if config['cpu'] ==1:
            device = torch.device('cpu')
            print("Using CPU")

        if config['cuda'] ==1 and torch.cuda.is_available():
            torch.manual_seed(config['seed'])
            torch.cuda.manual_seed_all(config['seed'])
            device = torch.device('cuda:0')
            print("Using CUDA with deterministic settings")
        
        if config['mps'] ==1 and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS")
        
        log_dir = os.path.expanduser(config['log_dir'])
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        torch.set_num_threads(8)

        envs = make_vec_envs(config['env_name'], config['seed'], config['num_processes'],
                            config['gamma'], config['log_dir'], device, False)

        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': config['recurrent_policy']}, pretrained_extractor=config['pretrained_encoder'])
        actor_critic.to(device)

        if config['pretrained_encoder']:
            print("Loading pretrained encoder weights")
            save_path = os.path.join('model_weights', 'representation_extraction_v2' + ".pth")
            self.load_encoder_weights(actor_critic, save_path, device)

        agent = algo.PPO(
            actor_critic,
            config['clip_param'],
            config['ppo_epoch'],
            config['num_mini_batch'],
            config['value_loss_coef'],
            config['entropy_coef'],
            lr=config['lr'],
            eps=config['eps'],
            max_grad_norm=config['max_grad_norm'])

        rollouts = RolloutStorage(config['num_steps'], config['num_processes'],
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size)

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)

        # Create CSV file
        csv_file = os.path.join(config['log_dir'], f"{config['save_name']}.csv")

        # Write header to CSV file
        header = ["Update", "Num Timesteps", "FPS", "Mean Reward", "Median Reward", "Min Reward", "Max Reward", "Dist Entropy", "Value Loss", "Action Loss"]
        with open(csv_file, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(header)

        # Access GCP bucket
        if config['upload_to_gcp']:
            self.bucket_blob_dir = f"{config['architecture_list'][config['architecture']]}/{config['save_name']}"
            gcp_key = config['gcp_key']
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=gcp_key
            storage_client = storage.Client()
            bucket = storage_client.bucket(config['gcp_bucket_name'])
            blob = bucket.blob(f"{self.bucket_blob_dir}/train_log.csv")
            blob.upload_from_filename(csv_file)
            # Reparse config to yaml
            config_yaml = yaml.dump(config)

            # Upload config to GCP
            blob_config = bucket.blob(f"{self.bucket_blob_dir}/settings.yaml")
            blob_config.upload_from_string(config_yaml)




        if config['load_model']:
            # Load model weights
            print(f"Loading model weights from {config['save_dir']}")
            save_path = os.path.join(config['save_dir'], config['algo'], config['load_weight_name'] + ".pth")
            self.load_model_weights(actor_critic, envs, save_path, device)

        start = time.time()
        num_updates = int(
            config['num_env_steps']) // config['num_steps'] // config['num_processes']
        
        with tqdm(total=num_updates, desc="Training Progress", unit="item") as pbar:
            for j in range(num_updates):

                if config['use_linear_lr_decay']:
                    # decrease learning rate linearly
                    utils.update_linear_schedule(
                        agent.optimizer, j, num_updates,
                        config['lr'])

                for step in range(config['num_steps']):
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])

                    # Obser reward and next obs
                    obs, reward, done, infos = envs.step(action)

                    for info in infos:
                        if 'episode' in info.keys():
                            episode_rewards.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                        for info in infos])
                    rollouts.insert(obs, recurrent_hidden_states, action,
                                    action_log_prob, value, reward, masks, bad_masks)

                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()

                rollouts.compute_returns(next_value, config['use_gae'], config['gamma'],
                                        config['gae_lambda'], config['use_proper_time_limits'])

                value_loss, action_loss, dist_entropy = agent.update(rollouts)

                rollouts.after_update()

                # save for every interval-th episode or for the last epoch
                if (j % config['save_interval'] == 0
                        or j == num_updates - 1) and config['save_dir'] != "":
                    save_path = os.path.join(config['save_dir'], config['save_name'])
                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass

                    torch.save([
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                    ], os.path.join(save_path, config['save_name'] +"_step_"+ str(j) + ".pt"))
                    
                    # Upload model weights to GCP
                    if config['upload_to_gcp']:
                        blob_model = bucket.blob(f"{self.bucket_blob_dir}/_step_{str(j)}.pt")
                        blob_model.upload_from_filename(os.path.join(save_path, config['save_name'] +"_step_"+ str(j) + ".pt"))

                if j % config['log_interval'] == 0 and len(episode_rewards) > 1:
                    total_num_steps = (j + 1) * config['num_processes'] * config['num_steps']
                    end = time.time()
                    pbar.set_postfix({"total_num_steps": total_num_steps, "mean_episode_rewards": np.mean(episode_rewards), "min_episode_rewards": np.min(episode_rewards), "max_episode_rewards": np.max(episode_rewards)})
                    # Write to CSV file
                    with open(csv_file, mode='a') as file:
                        writer = csv.writer(file)
                        writer.writerow([j, total_num_steps, int(total_num_steps / (end - start)), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards), dist_entropy, value_loss, action_loss])

                    # Upload updated CSV to GCP
                    if config['upload_to_gcp']:
                        blob.upload_from_filename(csv_file)

                if config['visualize'] and j % config['view_video_interval'] == 0 and len(episode_rewards) > 1:
                    for i in range(min(len(rollouts.obs),40)):
                        obs_copy = rollouts.obs[i].cpu().clone().detach().numpy()
                        images = []
                        if len(obs_copy.shape) ==4:
                            obs_copy = rollouts.obs[i].cpu().clone().detach().numpy()
                            image = None
                            for k in range (obs_copy.shape[0]):
                                observation_resized = cv2.resize(obs_copy[0,0,:,:], (200, 200))
                                observation_resized = cv2.cvtColor(observation_resized, cv2.COLOR_GRAY2RGB)
                                observation_resized = cv2.rectangle(observation_resized, (0, 0), (200, 200), (0, 0, 255), 2)
                                images.append(observation_resized)

                        else:
                            for k in range (obs_copy.shape[0]):
                                canvas = np.zeros((200, 200,3))
                                canvas = cv2.rectangle(canvas, (0, 0), (200, 200), (0, 0, 255), 2)
                                for j in range (obs_copy.shape[1]//2):
                                    x,y = (obs_copy[k][2*j]+2)/4, (obs_copy[k][2*j+1]+2)/4 
                                    if j == 0:
                                        cv2.circle(canvas, (int(x*200), int(y*200)), 20, (255, 255, 255), -1)
                                    elif j == 1:
                                        cv2.rectangle(canvas, (int((x-0.1)*200), int((y-0.1)*200)), (int((x+0.1)*200), int((y+0.1)*200)), (255, 255, 255), -1)
                                    else:
                                        cv2.line(canvas, (int((x)*200), int((y-0.1)*200)), (int((x+0.1)*200), int((y+0.1)*200)), (255, 255, 255), 2)
                                        cv2.line(canvas, (int((x)*200), int((y-0.1)*200)), (int((x-0.1)*200), int((y+0.1)*200)), (255, 255, 255), 2)
                                        cv2.line(canvas, (int((x-0.1)*200), int((y+0.1)*200)), (int((x+0.1)*200), int((y+0.1)*200)), (255, 255, 255), 2)
                                images.append(canvas)
                        
                        width = 4
                        height = ((obs_copy.shape[0])-1)//width +1
                        total_canvas = np.zeros((200*width, 200*height,3))

                        for i in range(len(images)):
                            x,y = i%width, (i)//width
                            image = images[i]
                            total_canvas[x*200:(x+1)*200, y*200:(y+1)*200,] = image

                        cv2.imshow('Environment', total_canvas)
                        cv2.waitKey(1)
                        time.sleep(0.1)

                if (config['eval_interval'] is not None and len(episode_rewards) > 1
                        and j % config['eval_interval'] == 0):
                    evaluate(actor_critic, None, config['env_name'], config['seed'],
                            config['num_processes']
                            , eval_log_dir, device)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    RL_main(config)
