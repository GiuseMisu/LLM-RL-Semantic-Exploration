""""
This file serves the purpose of testing a policy with a simple environment in order to see wether it works correctly
"""
import os
import warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")

import gymnasium as gym
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import sys
sys.path.append('../../../')  
from src.common.env_setup import make_minigrid_env
from ppo.ppo_config import PPO, RecurrentPPO


def save_frames_as_gif(frames, path='./', filename='Policy.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    plt.close()

def evaluate_policy(env, policy, n_episodes=10):
    """
    Evaluate the policy over multiple episodes and return statistics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        total_reward = 0
        state, _ = env.reset()
        done = trunc = False
        steps = 0
        
        while not done and not trunc:
            action, _ = policy.get_act(torch.FloatTensor(state).unsqueeze(0))
            env_action = F.softmax(action, dim=-1).argmax().item()
            state, reward, done, trunc, _ = env.step(env_action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
                
        print(f"Episode {episode+1}/{n_episodes}: Reward={total_reward:.2f}, Steps={steps}")
    
    # Calculate statistics
    stats = {
        'mean_reward': sum(episode_rewards) / n_episodes,
        'std_reward': (sum((r - sum(episode_rewards)/n_episodes)**2 for r in episode_rewards) / n_episodes)**0.5,
        'min_reward': min(episode_rewards),
        'max_reward': max(episode_rewards),
        'mean_length': sum(episode_lengths) / n_episodes,
        'all_rewards': episode_rewards
    }
    
    return stats

def play_agent_single_RUN(env, policy):
    total_reward = 0
    state, _ = env.reset()
    frames_gif=[]
    done = trunc = False
    while not done and not trunc:
        action, _ = policy.get_act(torch.FloatTensor(state).unsqueeze(0))
        env_action = F.softmax(action, dim=-1).argmax().item()
        new_state, reward, done, trunc, _ = env.step(env_action)
        total_reward += reward
        frames_gif.append(env.render())
        state = new_state

    save_frames_as_gif(frames_gif, filename = policy.name + ".gif")
    print("[SINGLE-RUN EVAL] Total Reward:", total_reward, "\n")    

def main():
    # old code
    # env_id = "LunarLander-v3"    
    # # Create the env
    # env = gym.make(env_id)
    # # Create the evaluation env
    # eval_env = gym.make(env_id, render_mode="rgb_array")


    env_id = "MiniGrid-DoorKey-5x5-v0"
    #env_id = "MiniGrid-Empty-16x16-v0" #"MiniGrid-Empty-8x8-v0" #MiniGrid-Empty-5x5-v0 IS TOO EASY


    # seed = 0 è quello facile con porta tutto sopra
    # seed = 1 é quello difficile con porta in mezzo /  
    seed = 0 

    # Create environment using your env_setup.py
    env = make_minigrid_env(env_id=env_id, 
                            render_mode="rgb_array", 
                            max_steps=150,
                            seed=seed)()
    
    print("\n=========== TRAINING PHASE===========\n")
    # Define the Policy
    policy = RecurrentPPO(env = env, 
                          # done automatically inside the code output_dim= 4, 
                          epochs = 25, 
                          gamma = 0.99, 
                          epsilon = 0.2,
                          encode_dim=128,  # CNN output
                          hidden_dim=64,    # LSTM hidden size
                          recurrence = "lstm",
                          model_name="doorkey_RecurrentPPO"
                          )

    # Train the environment
    policy.trainer(
        early_stopping_threshold = 0.95,  # average reward threshold for early stopping 
        window_size = 15  # Number of epochs to average over
        )    
    
    # load a trained version of the environment
    policy.load()

    print("\n\nEvaluating the trained policy")
    eval_env = make_minigrid_env(env_id=env_id, 
                                 render_mode="rgb_array", 
                                 max_steps=150
                                 )()

    # Evaluate the Environment
    policy.eval()
    
    # Evaluate over multiple episodes for statistics
    stats = evaluate_policy(eval_env, 
                            policy, 
                            n_episodes=10 # evaluation over 10 episodes
                            )
    
    print("\n" + "="*35)
    print("EVALUATION STATISTICS")
    print("="*35)
    print(f"Mean Reward:    {stats['mean_reward']:.3f} +/- {stats['std_reward']:.3f}")
    print(f"Min Reward:     {stats['min_reward']:.3f}")
    print(f"Max Reward:     {stats['max_reward']:.3f}")
    print(f"Mean Length:    {stats['mean_length']:.1f} steps")
    print("="*35)

    print("\nGenerating GIF")
    play_agent_single_RUN(eval_env, policy)

if __name__ == "__main__":
    main()