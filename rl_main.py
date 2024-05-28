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

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from tqdm import tqdm
import csv
import datetime

def load_model_weights(actor_critic, envs, save_path, device):
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        actor_critic.load_state_dict(checkpoint[0].state_dict())
        vec_norm = utils.get_vec_normalize(envs)
        if vec_norm is not None and checkpoint[1] is not None:
            vec_norm.obs_rms = checkpoint[1]
        print(f"Loaded model weights from {save_path}")
    else:
        print(f"No model weights found at {save_path}")

def load_encoder_weights(actor_critic, save_path, device):
    checkpoint = torch.load(save_path, map_location=device)
    if 'encoder_state_dict' in checkpoint:
        actor_critic.feature_extractor.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"Loaded encoder weights from {save_path}")
        print(actor_critic.feature_extractor.state_dict().keys())
    else:
        raise KeyError("Pretrained encoder weights not found in checkpoint")


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(8)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        #print("MPS is available")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        #print("CUDA is available")
    else:
        device = torch.device('cpu')
        #print("MPS is not available")

    print(device)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy}, pretrained_extractor=args.pretrained_encoder)
    actor_critic.to(device)


    if args.pretrained_encoder:
        print("Loading pretrained encoder weights")
        save_path = os.path.join('model_weights', 'representation_extraction_v2' + ".pth")
        load_encoder_weights(actor_critic, save_path, device)

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    #print(f"{obs.shape}")
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    # Create CSV file
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file = os.path.join(args.log_dir, f"{args.save_name}_{current_time}.csv")

    # Write header to CSV file
    header = ["Update", "Num Timesteps", "FPS", "Mean Reward", "Median Reward", "Min Reward", "Max Reward", "Dist Entropy", "Value Loss", "Action Loss"]
    with open(csv_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    
    if args.load_model:
        # Load model weights
        print(f"Loading model weights from {args.save_dir}")
        save_path = os.path.join(args.save_dir, args.algo, args.load_weight_name + ".pth")
        load_model_weights(actor_critic, envs, save_path, device)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in tqdm(range(num_updates), desc="Training Progress"):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
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

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.save_name +"_step_"+ str(j) + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
                        # Write to CSV file
            with open(csv_file, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([j, total_num_steps, int(total_num_steps / (end - start)), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards), dist_entropy, value_loss, action_loss])
            
       
        if j % args.view_video_interval == 0 and len(episode_rewards) > 1:
            for i in range(min(len(rollouts.obs),40)):
                obs_copy = rollouts.obs[i].cpu().clone().detach().numpy()
                images = []
                #print(obs_copy.shape)
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
                        # per processes
                        for j in range (obs_copy.shape[1]//2):
                            # per shape x y
                            #print(j)
                            
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

                    #print(x*200,(x+1)*200, y*200,(y+1)*200)
                    total_canvas[x*200:(x+1)*200, y*200:(y+1)*200,] = image


                cv2.imshow('Environment', total_canvas)
                cv2.waitKey(1)
                time.sleep(0.1)
            #cv2.destroyAllWindows()

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            evaluate(actor_critic, None, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    Environment_list = ["Sprites","SpritesState"]
    Distractors = ["-v0","-v1","-v2","-v3","-v4","-v5","-v6","-v7","-v8","-v9"]
    Policy_list = ["ORACLE","CNN_PPO","PRE_TRAINED"]
    Setting = [0,0,1]


    if Setting[0] == 1:
        env = Environment_list[Setting[0]]+Distractors[Setting[1]]
        Setting[2] = 0
        pre_trained = False
    else:
        env = Environment_list[Setting[0]]+Distractors[Setting[1]]
        pre_trained = (Setting[2] == 2)

    save_name = Policy_list[Setting[2]] + "_"+ str(Setting[1]) +"_distractors"

    if Setting[0]==1 and Setting[2] != 0:
        print("SpritesState only supports ORACLE policy")
        raise ValueError("Invalid Setting")
    if Setting[0]!=1 and Setting[2] == 0:
        print("Oracle policy only supports SpritesState environment")
        raise ValueError("Invalid Setting")
    
        
    



    args = get_args()
    args.env_name = env
    args.algo = "ppo"
    args.use_gae = True
    args.log_interval = 1
    args.num_steps = 200
    args.num_processes = 16
    args.seed = 100
    args.lr = 3e-4
    args.entropy_coef = 0
    args.value_loss_coef = 0.5
    args.ppo_epoch = 8
    args.num_mini_batch = 32
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.num_env_steps = 5000000
    args.use_linear_lr_decay = True
    args.use_proper_time_limits = True
    args.cuda = True
    args.log_dir = 'logs'
    args.save_dir = 'model_weights'
    args.save_weight_name = 'SpritesState-v0'
    args.load_model = False
    args.load_weight_name = 'SpritesState-v0'
    args.eval_interval = 100
    args.save_interval = 100
    args.view_video_interval = 20
    args.pretrained_encoder= pre_trained
    args.save_name = save_name
    main(args)
