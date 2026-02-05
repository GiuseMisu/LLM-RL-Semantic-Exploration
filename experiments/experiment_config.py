"""

Centralized hyperparameter configuration for all experiments

NOTE: 
reward function is defined R(t)=1 - 0.9⋅(t/max_steps), where t = steps to reach goal
so is not possible to achieve reward=1.0 -> early stopping threshold is set to 0.97 or below

"""

# Environment-specific settings (SAME for all methods in that env)
ENV_CONFIGS = {
    "empty_5x5": {
        "env_id": "MiniGrid-Empty-5x5-v0",
        "batch_size": 2048,
        "rollout_iterations": 4096,
        "max_steps": 250, #default
        "epochs": 50, #--> empty envs need less training
        "early_stopping_threshold": 0.96,
        "early_stopping_window": 10,
    },
    "empty_8x8": {
        "env_id": "MiniGrid-Empty-8x8-v0",
        "batch_size": 4096,
        "rollout_iterations": 16384,
        "max_steps": 640, #default 
        "epochs": 70, #--> empty envs need less training
        "early_stopping_threshold": 0.96,
        "early_stopping_window": 10,
    },
    "doorkey_5x5": {
        "env_id": "MiniGrid-DoorKey-5x5-v0",
        "batch_size": 2048,
        "rollout_iterations": 4096,
        "max_steps": 250, #default 
        "epochs": 200,
        "early_stopping_threshold": 0.96,
        "early_stopping_window": 10,
    },
    "doorkey_8x8": {
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "batch_size": 4096,
        "rollout_iterations": 16384, 
        "max_steps": 640, #default
        "epochs": 250,
        "early_stopping_threshold": 0.96,
        "early_stopping_window": 10,
    },
}

# Shared PPO parameters (SAME across all PPO-based methods)
SHARED_PPO_PARAMS = {
    "gamma": 0.99,
    "epsilon": 0.2,
    # already embedded inside the class "lr": 1e-3,
    # already embedded inside the class "entropy_coeff": 0.02,
}

RECURRENT_PARAMS = {
    "hidden_dim": 128,
    "sequence_length": 32,
    # already embedded inside the class "lr": 3e-4,  # RecurrentPPO uses lower learning rate
    # already embedded inside the class "entropy_coeff": 0.02,
    "encode_dim": 128,
    "recurrence": "lstm",
}

# Method-specific parameters
RND_PARAMS = {
    "gamma_intrinsic": 0.99,
    "intrinsic_reward_coeff": 0.005,
    "rnd_dim": 128,
}

LLM_PARAMS = {
    "llm_weight": 1.0,  # can vary in ablation studies
    "voting_samples": 3,
}

EUREKA_PARAMS = {
    "reflection_iterations": 3,
    "training_epochs": 50,
    "num_eval_episodes": 30,
}

EVALUATION_PARAMS = {
    "n_episodes": 20,
    "save_gif": True,
}