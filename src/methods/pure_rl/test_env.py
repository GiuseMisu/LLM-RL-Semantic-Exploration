""""
This file serves the purpose of testing a policy with a simple environment in order to see wether it works correctly
"""
import gymnasium as gym
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ppo.ppo_config import PPO


def save_frames_as_gif(frames, path='./', filename='Policy.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def play_agent(env, policy):
    total_reward = 0
    state, _ = env.reset()
    frames_gif=[]
    step = 0
    done = trunc = False
    while not done and not trunc:

        action, _ = policy(torch.tensor(state))
        env_action = F.softmax(action, dim=-1).argmax().item()
        new_state, reward, done, trunc, _ = env.step(env_action)
        total_reward += reward
        frames_gif.append(env.render())
        state = new_state
    save_frames_as_gif(frames_gif, filename = policy.name + ".gif")
    print("Total Reward:", total_reward)    

def main():
    env_id = "LunarLander-v3"
    
    # Create the env
    env = gym.make(env_id)

    # Create the evaluation env
    eval_env = gym.make(env_id, render_mode="rgb_array")

    # Define the Policy
    policy = PPO(env = env, input_dim = 8, output_dim= 4, epochs = 1000, gamma = 0.99, epsilon = 0.2)

    # Train the environment
    policy.trainer()
    
    # Evaluate the Environment
    policy.eval()
    
    play_agent(eval_env, policy)

if __name__ == "__main__":
    main()