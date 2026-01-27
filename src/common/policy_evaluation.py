from src.common.visualization import save_frames_as_gif

import torch
from torch.nn import functional as F

def evaluate_policy(env, policy, n_episodes=10, save_gif=True, gif_fps=10, gif_interval=100):
    """
    Evaluate the policy over multiple episodes, collect statistics and optionally save a GIF.
    Args:
        gif_fps: Frames per second for the GIF (higher = faster playback, 10=slow, 30=normal, 60=fast)
        gif_interval: Milliseconds between frames in animation (lower = faster, 100=slow, 50=normal, 20=fast)
    Returns:
        Dict with evaluation statistics
    """
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
    stats = {
        'mean_reward': sum(episode_rewards) / n_episodes,
        'std_reward': (sum((r - sum(episode_rewards)/n_episodes)**2 for r in episode_rewards) / n_episodes)**0.5,
        'min_reward': min(episode_rewards),
        'max_reward': max(episode_rewards),
        'mean_length': sum(episode_lengths) / n_episodes,
        'all_rewards': episode_rewards
    }
    
    print("\n" + "="*35)
    print("EVALUATION STATISTICS")
    print("="*35)
    print(f"Mean Reward:    {stats['mean_reward']:.3f} +/- {stats['std_reward']:.3f}")
    print(f"Min Reward:     {stats['min_reward']:.3f}")
    print(f"Max Reward:     {stats['max_reward']:.3f}")
    print(f"Mean Length:    {stats['mean_length']:.1f} steps")
    print("="*35)

    # Save GIF with all episodes
    if save_gif and all_frames:
        save_frames_as_gif(all_frames, filename=policy.name + "_eval.gif", 
                          fps=gif_fps, interval=gif_interval, episode_info=episode_info)
        
    return stats