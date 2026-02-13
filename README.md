# LLM-RL-Semantic-Exploration

A comparative study of reinforcement learning methods augmented with Large Language Models for solving sparse-reward navigation tasks in MiniGrid environments. The project benchmarks pure RL baselines (PPO, Recurrent PPO, RND) against two LLM-guided reward shaping strategies -- **Scalar Reward** and **Eureka** -- across environments of increasing complexity (Empty and DoorKey, from 5x5 to 16x16).


## Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Project Structure](#project-structure)
- [Environments](#environments)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Overview

Sparse-reward environments remain a core challenge in deep reinforcement learning. An agent receives a non-zero reward only upon reaching the goal, making credit assignment across long horizons extremely difficult. This project investigates whether LLMs can provide meaningful intermediate reward signals to accelerate learning.

Three families of approaches are implemented from scratch and compared:

1. **Pure RL** -- standard policy gradient methods operating on the raw environment reward.
2. **Curiosity-Driven RL** -- PPO augmented with Random Network Distillation (RND) for intrinsic exploration bonuses.
3. **LLM-Guided RL** -- two distinct strategies that leverage LLMs to shape the reward function.

To ensure a fair comparisons all methods share the same underlying PPO backbone, CNN encoder, and training infrastructure.


## Methods

### Pure RL Baselines

| Method | Description |
|---|---|
| **PPO** | Proximal Policy Optimization with a CNN encoder over 7x7x3 partial observations. Actor-critic architecture with clipped surrogate objective. |
| **Recurrent PPO** | PPO with an LSTM layer after the CNN encoder to handle partial observability. |
| **Curriculum PPO** | Progressive training across grid sizes (5x5 -> 6x6 -> 8x8 -> 16x16) with weight transfer and early-stopping promotion. |

### Curiosity-Driven

| Method | Description |
|---|---|
| **RND-PPO** | PPO with Random Network Distillation. Uses separate extrinsic and intrinsic value heads and advantage streams to encourage visiting novel states. |

### LLM-Guided Reward Shaping

#### Scalar Approach

The environment state is converted to a text description. At every step the LLM is prompted to return a scalar reward (between -0.1 and 1.0) reflecting the agent's progress. A caching layer with median voting (N samples) and physics-based guardrails filters out hallucinated rewards.

Supported LLM backends (via Ollama): Hermes-3 8B, DeepSeek-R1 8B.

#### Eureka Approach

Inspired by the [Eureka](https://arxiv.org/abs/2310.12931) paper. Instead of querying the LLM at every step, the LLM generates a complete Python reward function (`compute_reward(env)`). The system then:

1. Compiles and injects the generated function as a reward wrapper.
2. Trains a PPO agent with the shaped reward for a fixed number of epochs.
3. Evaluates performance and feeds metrics back to the LLM for reflection.
4. Repeats for K iterations, keeping the best reward function.

The best discovered reward function is saved and used for final extended training.

Supported LLM backends (via Ollama): DeepSeek-V3.1 671B, GPT-OSS 120B, Qwen3-Coder 480B.

#### Reward Function Transfer

Best reward functions discovered on smaller environments (e.g., DoorKey-8x8) can be transferred to larger ones (e.g., DoorKey-16x16) without re-running the search.


## Environments

All environments use the [MiniGrid](https://github.com/Farama-Foundation/Minigrid) library.

| Environment | Observation | Reward | Challenge |
|---|---|---|---|
| **Empty-5x5 / 16x16** | 7x7x3 partial grid | Sparse (goal only) | Navigation under partial observability |
| **DoorKey-5x5 / 8x8 / 16x16** | 7x7x3 partial grid | Sparse (goal only) | Multi-stage task: find key, unlock door, reach goal |

The reward function is defined as $R(t) = 1 - 0.9 \cdot (t / t_{\max})$ where $t$ is the number of steps to reach the goal.


## Project Structure

```
LLM-RL-Semantic-Exploration/
├── src/                              # Source code
│   ├── common/                       # Shared utilities
│   │   ├── env_setup.py              # Environment 
│   │   ├── metrics.py                # Logging 
│   │   ├── policy_evaluation.py      # N-episode evaluation 
│   │   └── visualization.py          # GIF animation
│   │
│   └── methods/
│       ├── pure_rl/                  # PPO, Recurrent PPO, Curriculum Learning
│       │   ├── ppo/                  # PPO implementation
│       │   ├── utils/                # Networks, rollout buffer, policy base class
│       │   └── curriculum_learning/  # Stage-based progressive trainer
│       │
│       ├── curiosity_driven/         # RND-PPO 
│       │
│       └── llm_guided/
│           ├── llm_clients/          # LLM client (Ollama)
│           ├── ScalarApproach/       # Per-step LLM reward with caching
│           └── EurekaApproach/       # Reward function generation
│
├── experiments/                      # Runnable experiment scripts
│   ├── empty_5x5/                 
│   ├── empty_16x16/
│   ├── doorkey_5x5/
│   ├── doorkey_8x8/
│   ├── doorkey_16x16/            
│   └── experiment_config.py          # Experiments hyperparameters
│
├── logs/                             # Training metrics (CSV) per experiment
├── results/
│   ├── models/                       # Saved model checkpoints (.pkl)
│   ├── reward_functions/             # Best LLM-generated reward functions (.py)
│   └── visualizations/               # Evaluation GIFs
│
├── analysis/                         # Plotting Comparison
├── requirements.txt
└── LICENSE
```


## Installation

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com/) (for LLM-guided methods).

```bash
git clone https://github.com/GiuseMisu/LLM-RL-Semantic-Exploration.git
cd LLM-RL-Semantic-Exploration
pip install -r requirements.txt
```

For LLM-guided experiments, pull the required models through Ollama:

```bash
ollama pull hermes3:8b
ollama pull deepseek-v3.1:671b-cloud
ollama pull gpt-oss:120b-cloud
ollama pull qwen3-coder:480b-cloud
```


## Usage

### Running an Experiment

Each experiment is a standalone script under `experiments/`. Examples:

```bash
# Pure PPO on DoorKey-8x8
python -m experiments.doorkey_8x8.run_ppo

# Recurrent PPO on DoorKey-16x16
python -m experiments.doorkey_16x16.run_recurrent_ppo

# RND-PPO on DoorKey-16x16
python -m experiments.doorkey_16x16.run_rnd

# Curriculum PPO (5x5 -> 16x16)
python -m experiments.doorkey_16x16.run_curriculum_ppo

# Eureka with Qwen on DoorKey-16x16
python -m experiments.doorkey_16x16.run_eureka_qwen

# LLM Scalar Reward with Hermes on DoorKey-8x8
python -m experiments.doorkey_8x8.run_scalar_hermes
```

### Hyperparameters

All hyperparameters are centralized in [experiments/experiment_config.py](experiments/experiment_config.py). Environment-specific settings (batch size, rollout iterations, epochs) and method-specific parameters (RND coefficients, Eureka reflection iterations, LLM voting samples) are defined there.



## Results

Training logs (CSV) are stored under `logs/`. Each run produces:

- **Metrics CSV** -- per-epoch training statistics (environment reward, loss, entropy, etc.).
- **Evaluation CSV** -- post-training evaluation over N episodes (mean reward, success rate, episode length).
- **GIFs** -- visual rollouts of the trained policy saved under `results/visualizations/`.

Best reward functions generated by the Eureka approach are saved as standalone Python files under `results/reward_functions/`.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
