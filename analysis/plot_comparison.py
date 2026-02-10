import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_env_reward(csv_paths: List[str], 
                    labels: Optional[List[str]] = None,
                    smooth_window: int = 10,
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Environment/Extrinsic Reward across experiments.
    Handles both 'Env_Reward' and 'Extrinsic_Reward' column names.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        
        # Handle both column names
        if 'Env_Reward' in df.columns:
            values = df['Env_Reward'].values
        elif 'Extrinsic_Reward' in df.columns:
            values = df['Extrinsic_Reward'].values
        else:
            print(f"Warning: Neither 'Env_Reward' nor 'Extrinsic_Reward' found in {label}")
            continue
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Environment Reward', fontsize=12)
    ax.set_title('Environment Reward Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_episode_length(csv_paths: List[str], 
                        labels: Optional[List[str]] = None,
                        smooth_window: int = 10,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Episode Length across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Episode_Length'].values
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)
    ax.set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
         
    
    plt.show()
    return fig


def plot_policy_loss(csv_paths: List[str], 
                     labels: Optional[List[str]] = None,
                     smooth_window: int = 10,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Policy Loss across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Policy_Loss'].values
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Policy Loss', fontsize=12)
    ax.set_title('Policy Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
         
    
    plt.show()
    return fig


def plot_value_loss(csv_paths: List[str], 
                    labels: Optional[List[str]] = None,
                    smooth_window: int = 10,
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Value Loss across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Value_Loss'].values
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Value Loss', fontsize=12)
    ax.set_title('Value Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_entropy(csv_paths: List[str], 
                 labels: Optional[List[str]] = None,
                 smooth_window: int = 10,
                 save_path: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Entropy across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Entropy'].values
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title('Entropy Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_success_rate(csv_paths: List[str],
                      labels: Optional[List[str]] = None,
                      smooth_window: int = 5,
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Success Rate across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Success_Rate'].values
        
        # Convert to percentage if needed
        if values.max() <= 1.0:
            values = values * 100
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    ax.set_xlabel('Reflection Iteration', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_key_pickup_rate(csv_paths: List[str],
                         labels: Optional[List[str]] = None,
                         smooth_window: int = 5,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Key Pickup Rate across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Key_Pickup_Rate'].values
        
        # Convert to percentage if needed
        if values.max() <= 1.0:
            values = values * 100
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
     
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    ax.set_xlabel('Reflection Iteration', fontsize=12)
    ax.set_ylabel('Key Pickup Rate (%)', fontsize=12)
    ax.set_title('Key Pickup Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_door_open_rate(csv_paths: List[str],
                        labels: Optional[List[str]] = None,
                        smooth_window: int = 5,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Door Open Rate across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Door_Open_Rate'].values
        
        # Convert to percentage if needed
        if values.max() <= 1.0:
            values = values * 100
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    ax.set_xlabel('Reflection Iteration', fontsize=12)
    ax.set_ylabel('Door Open Rate (%)', fontsize=12)
    ax.set_title('Door Open Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_mean_reward(csv_paths: List[str],
                     labels: Optional[List[str]] = None,
                     smooth_window: int = 5,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Mean Reward across experiments.
    """
    
    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))
    
    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Mean_Reward'].values
        
        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)
        
        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    ax.set_xlabel('Reflection Iteration', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Mean Reward Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_mean_steps(csv_paths: List[str],
                    labels: Optional[List[str]] = None,
                    smooth_window: int = 5,
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of Mean Steps across experiments.
    X-axis shows only integer ticks.
    """

    data_dict = {}
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        data_dict[label] = df

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(data_dict))

    for (label, df), color in zip(data_dict.items(), colors):
        epochs = df['epoch'].values
        values = df['Mean_Steps'].values

        ax.plot(epochs, values, alpha=0.3, color=color, linewidth=0.8)

        if len(values) >= smooth_window:
            smoothed = pd.Series(values).rolling(window=smooth_window, center=True).mean()
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(epochs, values, label=label, color=color, linewidth=2.5)

    # Force x-axis to integer ticks and format as integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    ax.set_xlabel('Reflection Iteration', fontsize=12)
    ax.set_ylabel('Mean Steps', fontsize=12)
    ax.set_title('Mean Steps Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


# Example usage
if __name__ == "__main__":
   
    import os
    import sys
    from pathlib import Path
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "logs"

    # empty5x5 = [
    #     str(logs_dir / "PPO_EMPTY_5x5" / "PPO_EMPTY_5x5_metrics.csv"),
    #     str(logs_dir / "RecurrentPPO_lstm_EMPTY_5x5" / "RecurrentPPO_lstm_EMPTY_5x5_metrics.csv"),
    #     str(logs_dir / "RNDPPO_EMPTY_5x5" / "RNDPPO_EMPTY_5x5_metrics.csv")
    # ]
    # label_empty5x5 = [
    #     "PPO",
    #     "Recurrent PPO",
    #     "RND PPO"
    # ]
    
    # -----------------------------     
    # empty16x16 = [
    #     str(logs_dir / "PPO_EMPTY_16x16" / "PPO_EMPTY_16x16_metrics.csv"),
    #     str(logs_dir / "RecurrentPPO_lstm_EMPTY_16x16" / "RecurrentPPO_lstm_EMPTY_16x16_metrics.csv"),
    #     str(logs_dir / "RNDPPO_EMPTY_16x16" / "RNDPPO_EMPTY_16x16_metrics.csv")
    # ]
    # label_empty16x16 = [
    #     "PPO",
    #     "Recurrent PPO",
    #     "RND PPO"
    # ]

    # -----------------------------    
    # doorkey5x5 = [
    #     str(logs_dir / "PPO_DOORKEY_5x5" / "PPO_DOORKEY_5x5_metrics.csv"),
    #     str(logs_dir / "RecurrentPPO_lstm_DOORKEY_5x5" / "RecurrentPPO_lstm_DOORKEY_5x5_metrics.csv"),
    #     str(logs_dir / "RNDPPO_DOORKEY_5x5" / "RNDPPO_DOORKEY_5x5_metrics.csv")
    # ]
    # label_doorkey5x5 = [ 
    #     "PPO",
    #     "Recurrent PPO",
    #     "RND PPO",
    # ]

    # -----------------------------    
    # doorkey8x8 = [
    #     str(logs_dir / "PPO_DOORKEY_8x8" / "PPO_DOORKEY_8x8_metrics.csv"),
    #     str(logs_dir / "RecurrentPPO_lstm_DOORKEY_8x8" / "RecurrentPPO_lstm_DOORKEY_8x8_metrics.csv"),
    #     str(logs_dir / "RNDPPO_DOORKEY_8x8" / "RNDPPO_DOORKEY_8x8_metrics.csv"),
    # ]
    # label_doorkey8x8 = [ 
    #     "PPO",
    #     "Recurrent PPO",
    #     "RND PPO",
    # ]

    # -----------------------------    
    #doorkey16x16 = [
    #      str(logs_dir / "PPO_DOORKEY_16x16" / "PPO_DOORKEY_16x16_metrics.csv"),
    #      str(logs_dir / "RecurrentPPO_lstm_DOORKEY_16x16" / "RecurrentPPO_lstm_DOORKEY_16x16_metrics.csv"),
    #      str(logs_dir / "RNDPPO_DOORKEY_16x16" / "RNDPPO_DOORKEY_16x16_metrics.csv"),
    #      str(logs_dir / "Eureka_DoorKey16x16_Qwen" / "Eureka_Qwen_DoorKey_16x16_PPO_FINAL_metrics.csv"),
    #      str(logs_dir / "Eureka_DoorKey16x16_GPT-OSS" / "Eureka_GPT-OSS_DoorKey_16x16_PPO_FINAL_metrics.csv"),
    #      str(logs_dir / "Eureka_DoorKey16x16_DeepSeek" / "Eureka_DeepSeek_DoorKey_16x16_PPO_FINAL_metrics.csv")
    # ]
    #label_doorkey16x16 = [
    #     "PPO",
    #     "Recurrent PPO",
    #     "RND PPO",
    #      "Qwen",
    #      "GPT-OSS",
    #      "DeepSeek"
    # ]

    # to_plot = doorkey16x16 # doorkey5x5 #empty16x16 #empty5x5 #doorkey16x16
    # to_label = label_doorkey16x16 # label_doorkey5x5 #label_empty16x16 #label_empty5x5 #label_doorkey16x16

    # # # create output directory and plot all metrics once using the full lists
    # output_dir = base_dir / "analysis" / "Eureka_DoorKey_16x16"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # plot_env_reward(to_plot, to_label, save_path=str(output_dir / "Env_Reward_DoorKey_16x16.png"))
    # plot_episode_length(to_plot, to_label, save_path=str(output_dir / "Episode_Length_DoorKey_16x16.png"))
    # plot_entropy(to_plot, to_label, save_path=str(output_dir / "Entropy_DoorKey_16x16.png"))

    #plot_policy_loss(doorkey8x8, label_doorkey8x8, save_path="policy_loss.png")
    #plot_value_loss(doorkey8x8, label_doorkey8x8, save_path="value_loss.png")
    

    Reflection_doorkey16x16 = [
         str(logs_dir / "Eureka_DoorKey16x16_Qwen" / "Eureka_Qwen_DoorKey_16x16_reflection.csv"),
         str(logs_dir / "Eureka_DoorKey16x16_GPT-OSS" / "Eureka_GPT-OSS_DoorKey_16x16_reflection.csv"),
         str(logs_dir / "Eureka_DoorKey16x16_DeepSeek" / "Eureka_DeepSeek_DoorKey_16x16_reflection.csv")
    ]
    label_reflection_doorkey16x16 = [
         "Qwen",
         "GPT-OSS",
         "DeepSeek"
    ]

    plot_success_rate(Reflection_doorkey16x16, label_reflection_doorkey16x16, save_path="Success_Rate_Reflection.png")
    plot_key_pickup_rate(Reflection_doorkey16x16, label_reflection_doorkey16x16, save_path="Key_Pickup_Rate_Reflection.png")
    plot_door_open_rate(Reflection_doorkey16x16, label_reflection_doorkey16x16, save_path="Door_Open_Rate_Reflection.png")
    plot_mean_reward(Reflection_doorkey16x16, label_reflection_doorkey16x16, save_path="Mean_Reward_Reflection.png")
    plot_mean_steps(Reflection_doorkey16x16, label_reflection_doorkey16x16, save_path="Mean_Steps_Reflection.png")
    
