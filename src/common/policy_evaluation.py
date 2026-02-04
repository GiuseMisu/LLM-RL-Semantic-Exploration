from src.common.visualization import save_frames_as_gif

import torch
from torch.nn import functional as F
import re
import os
import csv

def evaluate_policy(env, policy, n_episodes=10, save_gif=True, gif_fps=10, gif_interval=100):
    """
    Evaluate the policy over multiple episodes, collect statistics and optionally save a GIF.
    Args:
        gif_fps: Frames per second for the GIF (higher = faster playback, 10=slow, 30=normal, 60=fast)
        gif_interval: Milliseconds between frames in animation (lower = faster, 100=slow, 50=normal, 20=fast)
    Returns:
        Dict with evaluation statistics
    """

    env_type = "unknown"
    if hasattr(env.unwrapped, "spec") and env.unwrapped.spec is not None:
        env_id = env.unwrapped.spec.id
        size_match = re.search(r'(\d+)x(\d+)', env_id)
        if size_match:
            env_dimension = size_match.group(1) + 'x' + size_match.group(2)
            if "empty" in env_id.lower() or "minigrid-empty" in env_id.lower() :
                env_type = "EMPTY_" + env_dimension
            elif "door" in env_id.lower()  and "key" in env_id.lower()  or "doorkey" in env_id.lower() :
                env_type = "DOORKEY_" + env_dimension
            else:
                print(f"[WARNING] Unrecognized MiniGrid env type in env_id: {env_id}, defaulting to OTHER")
                env_type = "OTHER_" + env_dimension
        else:
            print(f"[WARNING] Could not parse env dimensions from env_id: {env_id}")
        

    episode_rewards = []
    episode_lengths = []
    all_frames = []  # Collect frames from all episodes
    episode_info = []  # Track which episode each frame belongs to
    
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
            
            # Collect frames for GIF
            if save_gif:
                all_frames.append(env.render())
                episode_info.append((episode + 1, n_episodes))  # Current episode, total episodes
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
                
        print(f"Episode {episode+1}/{n_episodes}: Reward={total_reward:.2f}, Steps={steps}")
    
    # Calculate statistics
    mean_reward = sum(episode_rewards) / n_episodes
    std_reward = (sum((r - mean_reward)**2 for r in episode_rewards) / n_episodes)**0.5
    mean_length = sum(episode_lengths) / n_episodes
    std_length = (sum((l - mean_length)**2 for l in episode_lengths) / n_episodes)**0.5
    success_rate = sum(1 for r in episode_rewards if r > 0) / n_episodes
    
    stats = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min(episode_rewards),
        'max_reward': max(episode_rewards),
        'mean_length': mean_length,
        'std_length': std_length,
        'success_rate': success_rate,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths
    }
    
    print("\n" + "="*50)
    print("EVALUATION STATISTICS")
    print("="*50)
    print(f"Mean Reward:    {stats['mean_reward']:.3f} +/- {stats['std_reward']:.3f}")
    print(f"Min Reward:     {stats['min_reward']:.3f}")
    print(f"Max Reward:     {stats['max_reward']:.3f}")
    print(f"Mean Length:    {stats['mean_length']:.1f} +/- {stats['std_length']:.1f} steps")
    print(f"Success Rate:   {stats['success_rate']:.1%} ({int(stats['success_rate'] * n_episodes)}/{n_episodes})")
    print("="*50)

    # Save evaluation metrics to CSV
    save_evaluation_to_csv(policy.name, env_type, stats, n_episodes)

    # Save GIF with all episodes
    if save_gif and all_frames:
        save_frames_as_gif(all_frames, filename=policy.name + "_" + env_type + "_eval.gif", 
                          fps=gif_fps, interval=gif_interval, episode_info=episode_info)
        
    return stats



def save_evaluation_to_csv(policy_name, env_type, stats, n_episodes):
    """
    Save evaluation metrics to a CSV file in the logs directory
    """
    
    csv_filename = f"{policy_name}_{env_type}_evaluation.csv" 
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, mode='a', newline='') as csv_file:
        fieldnames = [
            'n_episodes', 'mean_reward', 'std_reward', 
            'min_reward', 'max_reward', 'mean_length', 'std_length', 'success_rate'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write evaluation data
        writer.writerow({
            'n_episodes': n_episodes,
            'mean_reward': f"{stats['mean_reward']:.4f}",
            'std_reward': f"{stats['std_reward']:.4f}",
            'min_reward': f"{stats['min_reward']:.4f}",
            'max_reward': f"{stats['max_reward']:.4f}",
            'mean_length': f"{stats['mean_length']:.2f}",
            'std_length': f"{stats['std_length']:.2f}",
            'success_rate': f"{stats['success_rate']:.4f}"
        })
    
    print(f"Evaluation metrics saved to: {csv_filename}")