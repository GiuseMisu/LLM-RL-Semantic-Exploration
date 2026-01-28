import os
import warnings
# ---  SILENCE WARNINGS ---
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")


import re
import numpy as np
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from src.common.env_setup import make_minigrid_env
from src.methods.pure_rl.ppo.ppo_config import PPO  
from src.methods.llm_guided.EurekaApproach.eureka_prompt_doorkey import EUREKA_INITIAL_PROMPT_DOORKEY, EUREKA_FEEDBACK_PROMPT_TEMPLATE_DOORKEY

class EurekaSearch:
    def __init__(self, env_id, 
                 llm_model, reflection_iterations=5, 
                 evaluation_epochs=30, evaluation_max_steps=250, 
                 pure_rl_baseline='PPO'            
                 ):
        self.env_id = env_id
        self.llm_model = llm_model
        self.reflection_iterations = reflection_iterations
        self.best_code = None
        self.best_reward = -float('inf')

        self.evaluation_epochs = evaluation_epochs
        self.evaluation_max_steps = evaluation_max_steps
        self.pure_rl_baseline = pure_rl_baseline

    def clean_code_string(self, llm_output):
        # clean the llm ouput to preserve only the code part
        if llm_output is None:    
            print("[EUREKA-SEARCH ERROR] No code found in LLM output-> retunding empty string")
            return ""
        else:
            print(f"LLM Output:\n{llm_output}\n{'-'*40}")
            code_match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
            if code_match:
                print("the code had the python tags")
                return code_match.group(1).strip()
            if "def compute_reward" in llm_output:
                print("the code had no python tags but contains 'def compute_reward'")
                return llm_output
        

    # def evaluate_candidate(self, reward_code):
    #     """
    #     quality check of the current python function returned by the LLM
    #     Runs a QUICK training session check if the agent learns
    #     if not learning ask the llm to improve the code
    #     """
    #     try:
    #         env = make_minigrid_env(
    #             env_id=self.env_id,
    #             render_mode="rgb_array", 
    #             eureka_reward_code=reward_code,
    #             use_llm_rewards=False, 
    #             max_steps=self.evaluation_max_steps  # Short horizons for quick checks
    #         )()

    #         policy = PPO(
    #             env=env,
    #             gamma=0.99,
    #             epsilon=0.2,
    #             epochs=self.evaluation_epochs,  # SHORT TRAINING just to see slope
    #             model_name=self.pure_rl_baseline
    #         )

    #         # policy.batch_size = 4096
    #         # policy.rollout.iterations = 8192
    #         print(f"[Eureka] Training policy with: batch_size={policy.batch_size}, rollout.iterations={policy.rollout.iterations}")
                        
    #         # 3. Train
    #         policy.trainer()
                        
    #         # =========================================================================
    #         # 4. Extract Stats to Evaluate Reward Function Quality --> CRITIC ASPECT
    #         # =========================================================================
    #         # in this section evaluate the agent performance after training
    #         # to udnerstand wether the reward function is good or not
    #         # how to check if the agent is learning?
    #         #
    #         # - read some logs/stats from the training?
    #         #
    #         # - run a few eval episodes and see if rewards improved 
    #         #   =>Problem how to estimate improvement? 
    #         #      - based on the reward obtained?       
    #         #      - based on how far from the goal it gets?
    #         #      - based on success rate of solving env?
    #         #      - based on steps to solve? (has the key? has passed door? has reached goal?)
    #         # =========================================================================

    #         # let's run a quick eval episode
    #         total_rewards = []
    #         successes = 0
    #         eval_episode_num = 5
    #         print("====[EVALUATION OF LLM_REWARD_FUNCTION]====")
    #         for _ in range(eval_episode_num):
    #             obs, _ = env.reset()
    #             done = False
    #             ep_reward = 0
    #             while not done:
    #                 action = policy.get_action(obs)
    #                 obs, reward, terminated, truncated, _ = env.step(action)
    #                 done = terminated or truncated
    #                 ep_reward += reward
    #                 print(f"LLM-Reward: {reward}")
                    
    #                 #==CRITICAL==
    #                 # NON PENSO Vada bene perché reward restituita solo se lo risolve
    #                 # con un train cosi basso come puó'risolverlo?
    #                 # SE MINIGRID É PIÚ GRANDE DI 5X5 COME CAMBIA?
    #                 if reward > 0.5: 
    #                      successes += 1

    #                 # Check if episode ended due to reaching goal (terminated=True, not truncated)
    #                 # same problem of before
    #                 # if terminated and not truncated:
    #                 #     successes += 1

    #             total_rewards.append(ep_reward)

    #         mean_reward = np.mean(total_rewards)
    #         success_rate = successes / eval_episode_num
    #         return mean_reward, success_rate, None

    #     except Exception as e:
    #         print(f"[Eureka Search Error] Evaluation failed: {e} -> retunrning -inf reward")
    #         return -float('inf'), 0.0, str(e)

    #=====================================================
    #       EVALUATE HEURISTIC PYTHON FUNCTION V2
    #=====================================================
    def evaluate_candidate_v2(self, reward_code, num_eval_episodes=10):
        """
        Evaluates reward function quality through:
        2. Task completion metrics (success rate, key pickup, door opening)
        3. Behavioral metrics (exploration coverage)
        """       

        try:
            # Setup environment with Eureka reward
            env = make_minigrid_env(
                env_id=self.env_id,
                render_mode="rgb_array", 
                eureka_reward_code=reward_code,
                use_llm_rewards=False, 
                max_steps=self.evaluation_max_steps  # Short horizons for quick checks                 
            )()

            policy = PPO(
                env=env,
                gamma=0.99,
                epsilon=0.2,
                epochs=self.evaluation_epochs,  # SHORT TRAINING just to see slope
                model_name=self.pure_rl_baseline
            )
            
            reward_delta = 0.0
            try:
                training_history = policy.trainer(
                                                early_stopping_threshold=None # avoid early stopping during evaluation
                                                )

                # Analyze training history for reward improvement
                if training_history and len(training_history) > 1:
                    # Compare average of last 3 epochs vs first 3 epochs
                    start_perf = np.mean(training_history[:3])
                    end_perf = np.mean(training_history[-3:])
                    reward_delta = end_perf - start_perf
                else:
                    reward_delta = 0.0

            except Exception as train_err:
                print(f"[Eureka] Training failed with error: {train_err}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to be caught by outer try-except
      
            # ====================
            # EVALUATION METRICS
            # ====================

            #----------------metric 1----------------
            # MODIFY THE TRAININER FUNCTION TO return episode rewards during training
            # AND SEE IF THERE IS A SLOPE IN THE REWARD ACCUMULATED OR NOT
            #----------------------------------------

            # Metric 2: Task Completion Metrics
            eval_stats = {
                'successes': 0,
                'key_pickups': 0,
                'door_opens': 0,
                'total_rewards': [],
                'steps_to_success': []
            }
                        
            print("====[EVALUATION OF LLM_REWARD_FUNCTION]====")
            for ep_idx in range(num_eval_episodes):
                obs, _ = env.reset()
                done = False
                ep_reward = 0
                steps = 0
                
                # Track task progress
                picked_up_key = False
                opened_door = False                
                while not done:
                    # this version does not work due to -> .to() function in PPO.forward
                    # action, _ = policy.get_act(obs)
                    # obs, reward, terminated, truncated, info = env.step(action)

                    # Robustly get device, convert numpy obs -> tensor, add batch dim
                    device = getattr(policy, "device", next(policy.parameters()).device)
                    state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    # Policy output (could be logits, action tensor, or (action, value))
                    policy_out = policy.get_act(state)

                    # Normalize to single tensor that represents action/logits
                    if isinstance(policy_out, tuple):
                        out_tensor = policy_out[0]
                    else:
                        out_tensor = policy_out

                    # If output logits/probs (last dim > 1) -> argmax, else treat as action index
                    if isinstance(out_tensor, torch.Tensor):
                        # If shape is [Batch, Actions] (e.g., [1, 7]), it's logits -> use argmax
                        if out_tensor.dim() >= 2 and out_tensor.shape[-1] > 1:
                            action_tensor = torch.argmax(out_tensor, dim=-1)
                        else:
                            # It's already an index
                            action_tensor = out_tensor.squeeze()
                        
                        arr = action_tensor.detach().cpu().numpy().reshape(-1) # Safe conversion to python int
                        action = int(arr[0])
                    else:
                        # Fallback if policy returns a raw int/float
                        action = int(out_tensor)

                    # Step
                    obs, reward, terminated, truncated, info = env.step(action)

                    done = terminated or truncated
                    ep_reward += reward
                    steps += 1
                    
                    # Check task SUBGOAL (access unwrapped env)
                    # SUBGOAL 1: Picked up key / 2: Opened door / 3: Reached goal
                    unwrapped_env = env.unwrapped
                    if unwrapped_env.carrying is not None and unwrapped_env.carrying.type == 'key':
                        picked_up_key = True
                    
                    # Check if door exists and is open
                    for i in range(unwrapped_env.grid.width):
                        for j in range(unwrapped_env.grid.height):
                            cell = unwrapped_env.grid.get(i, j)
                            if cell and cell.type == 'door' and cell.is_open:
                                opened_door = True
                    
                    # Success: reached goal (terminated, not truncated)
                    if terminated and not truncated:
                        eval_stats['successes'] += 1
                        eval_stats['steps_to_success'].append(steps)
                        break
                
                if picked_up_key:
                    eval_stats['key_pickups'] += 1
                if opened_door:
                    eval_stats['door_opens'] += 1
                eval_stats['total_rewards'].append(ep_reward)
            print(f"  Eval ep {ep_idx+1}: reward={ep_reward:.2f}, steps={steps}, key={picked_up_key}, door={opened_door}")
        
            # Calculate final metrics
            success_rate = eval_stats['successes'] / num_eval_episodes
            key_pickup_rate = eval_stats['key_pickups'] / num_eval_episodes
            door_open_rate = eval_stats['door_opens'] / num_eval_episodes
            mean_reward = np.mean(eval_stats['total_rewards'])
            mean_steps_to_success = np.mean(eval_stats['steps_to_success']) if eval_stats['steps_to_success'] else float('inf')
            
            # =====================================================================
            # AGGREGATE SCORE (combine metrics into single fitness score)
            # =====================================================================
            # Weight different aspects of learning
            fitness_score = (
                success_rate * 100 +           # Most important is the goal reached
                key_pickup_rate * 20 +         # Partial credit for subtasks
                door_open_rate * 50 +
                (1.0 / (mean_steps_to_success + 1)) * 10  # Efficiency bonus
            )
            
            return {
                'fitness_score': fitness_score,
                'success_rate': success_rate,
                'key_pickup_rate': key_pickup_rate,
                'door_open_rate': door_open_rate,
                'mean_reward': mean_reward,
                'mean_steps': mean_steps_to_success,
                'reward_delta': reward_delta
            }, None  # No error
            
        except Exception as e:
            print(f"\n\n[Eureka Search Error] Evaluation failed with exception:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            
            return {
                'fitness_score': -float('inf'),
                'success_rate': 0.0,
                'key_pickup_rate': 0.0,
                'door_open_rate': 0.0,
                'mean_reward': -float('inf'),
                'mean_steps': float('inf'),                
                'reward_delta': reward_delta
            }, str(e)


    def run(self):
        current_prompt = EUREKA_INITIAL_PROMPT_DOORKEY
        
        for i in range(self.reflection_iterations):
            print(f"\n>>> Iteration to improve HEURISTIC: {i+1}/{self.reflection_iterations}")
            
            # 1. Generate
            # no need to pass through the cache in this approach so call irectly llm_model._get_raw_response
            response = self.llm_model._get_raw_response(current_prompt, 
                                                        False # it is the generate_explanation parameter, not used by DeepSeek but it must be passed
                                                        )

            code = self.clean_code_string(response)
            
            if not code:
                print(f"[Eureka Search] iteration:{i}/{self.reflection_iterations} No valid code generated, skipping evaluation.")
                continue
                
            # 2. Evaluate
            #----evaluuate v1
            #mean_rew, success, err = self.evaluate_candidate(code)
            #print(f"Stats -> Mean Reward: {mean_rew:.3f}, Success: {success:.2f}")

            #----evaluate v2
            eval_stats, err = self.evaluate_candidate_v2(code)
            print(f"Stats \n-> {eval_stats}\n")
            mean_rew = eval_stats['mean_reward'] # the mean of the total_summed_reward per episode
            success = eval_stats['success_rate']

            # 3. Update Best just to check
            if mean_rew > self.best_reward:
                self.best_reward = mean_rew
                self.best_code = code
                print("\n--- New Best Reward Function ---\n")
                
                if "deepseek-r1" in self.llm_model.model_name or "deepseek-r1:8b" in self.llm_model.model_name:
                    model_name = "DeepSeek_R1_8b"
                elif "deepseek-v3.1" in self.llm_model.model_name or "deepseek-v3" in self.llm_model.model_name or "671b-cloud" in self.llm_model.model_name:
                    model_name = "DeepSeek_671b"
                elif "phi3.5" in self.llm_model.model_name or "phi3_5" in self.llm_model.model_name or self.llm_model.model_name.startswith("phi3"):
                    model_name = "Phi3_5"
                else:
                    model_name = "UnknownModel"

                if "5x5" in self.env_id and "DoorKey" in self.env_id:
                    name_env = "DoorKey_5x5"
                elif "8x8" in self.env_id and "DoorKey" in self.env_id:
                    name_env = "DoorKey_8x8"
                elif "5x5" in self.env_id and "Empty" in self.env_id:
                    name_env = "Empty_5x5"
                elif "8x8" in self.env_id and "Empty" in self.env_id:
                    name_env = "Empty_8x8"
                else:
                    name_env = "UnknownEnv"

                with open(f"best_RwdFunc_{name_env}_{model_name}.py", "w") as f:
                    f.write(code)

            # 4. Feedback
            # V1 VERSION feedback_text = "The agent is stuck." if success < 0.1 else "Good progress."

            # V2 VERSION
            # create a dettailed feedback based on the eval stats to add to the feedback prompt
            feedback_text = self._generate_feedback(eval_stats)

            previous_code = code
            #if pyhton badge already present do not add again
            if "```python" not in previous_code:
                previous_code = f"\n```python\n{code}\n```"
            else:
                previous_code = f"\n{code}\n"

            current_prompt = EUREKA_FEEDBACK_PROMPT_TEMPLATE_DOORKEY.format(
                #1. the previous code submitted must be the first thing in the prompt 
                previous_code=code, 
                
                # 2. only after the previous code add the part with metrics analysis
                success_rate=eval_stats['success_rate'] * 100,
                key_pickup_rate=eval_stats['key_pickup_rate'] * 100,  
                door_open_rate=eval_stats['door_open_rate'] * 100,    
                mean_steps=eval_stats['mean_steps'],                  
                mean_reward=eval_stats['mean_reward'],

                #3. error log if any
                error_log=err,
                feedback_text=feedback_text
            )

            print("- "*20)
            print("New Feedback prompt for LLM")
            print(current_prompt)
            print("-"*20)
            
    def _analyze_reward_slope(self, delta, success_rate):
        """Analyzes if the agent was learning, unlearning, or stagnant."""
        # If the agent is already solving the task, the slope matters less
        if success_rate > 0.1:
            return "" 
            
        if delta > 0.5:
            return "PROMISING: The agent is LEARNING (positive reward slope during training) but ran out of time. The reward shaping is likely valid but too weak."
        elif delta < -0.5:
            return "WARNING: The agent is UNLEARNING (negative reward slope). The reward function might be encouraging suicidal or cyclic behavior."
        elif abs(delta) < 0.1:
            return "STAGNANT: The agent learned NOTHING (flat reward slope). The reward is too sparse or gradients are zero."
        
        return ""

    def _generate_feedback(self, metrics):
        """Generate stage-specific diagnostic feedback"""
        feedback = []

        # 1. Insert the Slope Analysis at the very top of feedback
        slope_msg = self._analyze_reward_slope(metrics.get('reward_delta', 0.0), metrics['success_rate'])
        if slope_msg:
            feedback.append(slope_msg)
        
        # --- CHECK FOR REWARD HACKING ---
        # Heuristic: High Reward but Low Success implies the agent is farming points
        # Threshold: If mean_reward is > higher then arbitrary high number but success is < 10%
        if metrics['mean_reward'] > 5.0 and metrics['success_rate'] < 0.1:
            feedback.append(
                f"CRITICAL WARNING - REWARD HACKING DETECTED: "
                f"The agent accumulated a massive reward ({metrics['mean_reward']:.2f}) but failed to solve the task. "
                "The agent is likely exploiting a bug (e.g., wiggling back and forth or toggling a switch repeatedly). "
                "FIX: Ensure rewards for repetitive actions (like distance or toggling) are strictly limited or potential-based."
            )
            return "\n".join(feedback) # Return immediately, this is the priority fix
        
        # --- Stage 1: Key Pickup ---
        if metrics['key_pickup_rate'] < 0.3:
            feedback.append(
                "CRITICAL: Agent rarely picks up the key (<30%). "
                "The reward function likely fails to guide the agent toward the key. "
                "Consider: (1) stronger distance-based shaping to key location, "
                "(2) substantial bonus for picking up key, "
                "(3) check if key location detection works correctly."
            )
        elif metrics['key_pickup_rate'] < 0.7:
            feedback.append(
                "MODERATE: Agent sometimes picks up key (30-70%). "
                "Reward shaping toward key is partially working but unstable. "
                "Consider: increasing key pickup reward or smoothing distance rewards."
            )
        else:
            feedback.append("GOOD: Agent consistently picks up key (>70%).")
        
        # --- Stage 2: Door Opening ---
        if metrics['key_pickup_rate'] > 0.7 and metrics['door_open_rate'] < 0.3:
            feedback.append(
                "CRITICAL: Agent picks up key but rarely opens door (<30%). "
                "The reward function fails to guide the agent from key to door. "
                "Consider: (1) distance-based reward to door AFTER key pickup, "
                "(2) substantial bonus for opening door, "
                "(3) check door detection logic."
            )
        elif metrics['door_open_rate'] > 0.7:
            feedback.append("GOOD: Agent consistently opens door (>70%).")
        
        # --- Stage 3: Goal Reaching ---
        if metrics['door_open_rate'] > 0.7 and metrics['success_rate'] < 0.3:
            feedback.append(
                "CRITICAL: Agent opens door but rarely reaches goal (<30%). "
                "The reward function fails to guide from door to goal. "
                "Consider: (1) very strong distance-based reward to goal after door opens, "
                "(2) massive bonus for reaching goal, "
                "(3) ensure goal detection works."
            )
        elif metrics['success_rate'] > 0.5:
            feedback.append(
                f"EXCELLENT: High success rate ({metrics['success_rate']*100:.1f}%). "
                f"Now optimize for efficiency - current average {metrics['mean_steps']:.0f} steps."
            )
        
        # --- Efficiency analysis ---
        if metrics['success_rate'] > 0.3 and metrics['mean_steps'] > 100:
            feedback.append(
                "EFFICIENCY ISSUE: Agent solves task but takes too many steps. "
                "Consider: adding small negative step penalty or stronger direct-path rewards."
            )
        
        # --- Overall assessment ---
        if not feedback:
            feedback.append(
                "STAGNATION: No clear progress on any task stage. "
                "The reward function may be too sparse or have conflicting signals. "
                "Consider: complete redesign focusing on dense distance-based shaping."
            )
        
        return "\n".join(feedback)