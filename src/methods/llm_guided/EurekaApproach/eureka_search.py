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
import traceback

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from src.common.env_setup import make_minigrid_env
from src.methods.pure_rl.ppo.ppo_config import PPO  
from src.methods.llm_guided.EurekaApproach.eureka_prompt_doorkey import EUREKA_INITIAL_PROMPT_DOORKEY, EUREKA_FEEDBACK_PROMPT_TEMPLATE_DOORKEY, MINIGRID_API_CONTEXT_DOORKEY

class EurekaSearch:
    def __init__(self, env_id, 
                 llm_model, 
                 reflection_iterations=5, # numero di volte refine prompt and reward function
                 
                 training_epochs=30, train_max_steps=250,  # training params pre eval
                 num_eval_episodes=10, # numero di episodi su cui valuti post training
                 pure_rl_baseline='PPO'            
                 ):
        self.env_id = env_id

        #extract the env size to place them in the dynamic context prompt
        if "5x5" in env_id:
            self.env_width = 5
            self.env_height= 5 
        elif "8x8" in env_id:
            self.env_width = 8
            self.env_height= 8
        else:
            #find number in the env id string and extract it
            size_match = re.search(r'(\d+)x(\d+)', env_id)
            if size_match:
                self.env_width = int(size_match.group(1))
                self.env_height= int(size_match.group(2))
            else:
                raise ValueError(f"Cannot extract env size from env_id: {env_id}")

        # the LLM client is based on the standard implementation defined in the relative file
        # currently interaction with LLM is purely STATELESS (i.e. no conversation memory)
        # keep a chat history, the LLM will see N different versions of compute_reward often mixes them up
        # we do manually formatting the prompt with {previous_code} and {feedback}, this ensures LLM pays attention to the exact thing we want it to fix, rather than getting lost in a long conversation log.
        # PROOF:
        # this approach is confirmed byt the eureka paper that talks about Evolutionary Algorithm, not a Conversational Agent
        # Moreover the Algorithm 1 showes that each time they query the LLM for reflection thay add to the prompt the previous code, this would not be necessary in a conversational agent that has memory of past messages
        self.llm_model = llm_model      
          
        self.best_code = None
        self.best_reward = -float('inf')

        # numero di volte refine prompt and reward function
        self.reflection_iterations = reflection_iterations

        # parameters regarding the training phase before the evaluation
        self.training_epochs = training_epochs
        self.train_max_steps = train_max_steps

        # evaluation params
        self.num_eval_episodes = num_eval_episodes

        self.pure_rl_baseline = pure_rl_baseline

    def clean_code_string(self, llm_output):
        # clean the llm ouput to preserve only the code part
        if llm_output is None:    
            print("[EUREKA-SEARCH ERROR] No code found in LLM output-> retunding empty string")
            return ""
        else:
            #[debug] print the function generated
            #print(f"LLM Output:\n{llm_output}\n{'-'*40}")
            # old code_match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
            code_match = re.search(r"```[Pp]ython\s*(.*?)```", llm_output, re.DOTALL)
            if code_match:
                print("the code had the python tags")
                return code_match.group(1).strip()
            else:
                if "def compute_reward" in llm_output:
                    print("the code had no python tags but contains 'def compute_reward'")
                    return llm_output
                else:
                    print("[EUREKA-SEARCH ERROR] No valid code block found in LLM output-> retunding empty string")
                    return ""
        
    #=====================================================
    #       EVALUATE HEURISTIC PYTHON FUNCTION V2
    #=====================================================
    def evaluate_candidate_v2(self, reward_code):
        """
        Evaluates reward function quality through both:
        - Training metrics (reward improvement during training) 
            -> less informative more noise takes random action to explore so may not reflect the reward function quality
        - Evaluation metrics (success rate, key pickup, door opening)
            -> more robust agent freezed from learning and tested on multiple different episodes
        """       
        train_reward_delta = 0.0  # Initialize early to avoid UnboundLocalError in exception handler
        env = None  # Initialize for finally block
        
        try:
            # Setup environment with Eureka reward
            env = make_minigrid_env(
                env_id=self.env_id,
                render_mode="rgb_array", 
                eureka_reward_code=reward_code,
                use_llm_rewards=False, 
                max_steps=self.train_max_steps  # Short horizons for quick checks                 
            )()

            policy = PPO(
                env=env,
                gamma=0.99,
                epsilon=0.2,
                epochs=self.training_epochs,  # SHORT TRAINING just to see slope
                model_name=self.pure_rl_baseline,

                save_pkl_model=False,  # Do not save model during Eureka evaluations
                track_stats=False  # Do not track detailed stats to save time
            )

            policy.batch_size = 1024 #2048  # 4096 for 8x8 / 2048 # for 5x5
            # rollout buffer size to match or exceed the batch size
            policy.rollout.iterations = 2048 #4096  # for 8x8 16384 / # for 5x5 4096
            
            print("\n====[TRAINING WITH LLM_REWARD_FUNCTION]====\n")
            try:
                training_history = policy.trainer(
                                                early_stopping_threshold=None # avoid early stopping during evaluation
                                                )
                
                # ====================
                # TRAINING METRICS               
                # Analyze training history for reward improvement
                # ====================
                if training_history and len(training_history) > 1:
                    # Compare average of last 3 epochs vs first 3 epochs
                    start_perf = np.mean(training_history[:3])
                    end_perf = np.mean(training_history[-3:])
                    train_reward_delta = end_perf - start_perf
                else:
                    train_reward_delta = 0.0

            except Exception as train_err:
                print(f"[Eureka] Training failed with error: {train_err}")                
                traceback.print_exc()
                raise  # Re-raise to be caught by outer try-except 
      
            # ====================
            # EVALUATION METRICS
            # ====================
            eval_stats = {
                'successes': 0,
                'key_pickups': 0,
                'door_opens': 0,
                'total_rewards': [],
                'steps_to_success': []
            }
                        
            print("\n\n====[EVALUATION OF LLM_REWARD_FUNCTION]====")
            print(f"Running {self.num_eval_episodes} evaluation episods to check the quality of the LLM-generated RwdFunc")
            for ep_idx in range(self.num_eval_episodes):
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
                    # device = getattr(policy, "device", next(policy.parameters()).device)
                    # state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    # # Policy output (could be logits, action tensor, or (action, value))
                    # policy_out = policy.get_act(state)

                    # # Normalize to single tensor that represents action/logits
                    # if isinstance(policy_out, tuple):
                    #     out_tensor = policy_out[0]
                    # else:
                    #     out_tensor = policy_out

                    # # If output logits/probs (last dim > 1) -> argmax, else treat as action index
                    # if isinstance(out_tensor, torch.Tensor):
                    #     # If shape is [Batch, Actions] (e.g., [1, 7]), it's logits -> use argmax
                    #     if out_tensor.dim() >= 2 and out_tensor.shape[-1] > 1:
                    #         action_tensor = torch.argmax(out_tensor, dim=-1)
                    #     else:
                    #         # It's already an index
                    #         action_tensor = out_tensor.squeeze()
                        
                    #     arr = action_tensor.detach().cpu().numpy().reshape(-1) # Safe conversion to python int
                    #     action = int(arr[0])
                    # else:
                    #     # Fallback if policy returns a raw int/float
                    #     action = int(out_tensor)

                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(policy.device)
                    with torch.no_grad():
                        action_logits, _ = policy.get_act(state_tensor)
                        action_prob = torch.nn.functional.softmax(action_logits, dim=-1)
                        dist = torch.distributions.Categorical(action_prob)
                        
                        # Use SAMPLING (same as training) - greedy fails on stochastic policies
                        action = dist.sample().item()

                    # Step
                    obs, reward, terminated, truncated, info = env.step(action)

                    # check if in the evaluation the goal was reached
                    if terminated and not truncated:
                        print(f"\t\t[Evaluation Phase] ep:{ep_idx+1} reached the goal")

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
                print(f"  Eval ep {ep_idx+1}: reward={ep_reward:.2f}, key={picked_up_key}, door={opened_door}", flush=True)
                
            # Calculate final metrics
            success_rate = eval_stats['successes'] / self.num_eval_episodes
            key_pickup_rate = eval_stats['key_pickups'] / self.num_eval_episodes
            door_open_rate = eval_stats['door_opens'] / self.num_eval_episodes
            mean_reward = np.mean(eval_stats['total_rewards'])
            mean_steps_to_success = np.mean(eval_stats['steps_to_success']) if eval_stats['steps_to_success'] else float('inf')
            
            # =====================================================================
            # AGGREGATE SCORE (combine metrics into single fitness score)
            # =====================================================================
            # Weight different aspects of learning
            fitness_score = (
                success_rate * 200 +           # Most important is the goal reached
                key_pickup_rate * 30 +         # Partial credit for subtasks
                door_open_rate * 70 +
                max(0, mean_reward) * 50 +
                (1.0 / (mean_steps_to_success + 1)) * 10 +  # Efficiency bonus
                train_reward_delta * 30           # Was training improving?
            )
            
            return {
                'fitness_score': fitness_score,
                'success_rate': success_rate,
                'key_pickup_rate': key_pickup_rate,
                'door_open_rate': door_open_rate,
                'mean_reward': mean_reward,
                'mean_steps': mean_steps_to_success,
                'train_reward_delta': train_reward_delta
            }, None  # No error
            
        except Exception as e:            
            full_error_trace = traceback.format_exc()            
            
            return {
                'fitness_score': -float('inf'),
                'success_rate': 0.0,
                'key_pickup_rate': 0.0,
                'door_open_rate': 0.0,
                'mean_reward': -float('inf'),
                'mean_steps': float('inf'),                
                'train_reward_delta': train_reward_delta
            }, full_error_trace
        
        finally:
            # Clean up environment to prevent memory leaks
            if env is not None:
                env.close()

    
    def _analyze_general_trend(self, delta, success_rate, mean_reward):
        """
        Combines Delta (Learning) and Success (Performance) to diagnose the agent state.
        Returns a high-priority string if a major issue is found.
        """
        # 1. Reward Hacking cHECK
        # If the agent is learning rapidly (High Delta) but failing in validation (Zero Success)
        # OR if the agent has accumulated massive rewards (High Mean) with Zero Success in Eval
        if (delta > 0.1 or mean_reward > 5.0) and success_rate < 0.05:
            return (
                f"CRITICAL WARNING - MISALIGNED OBJECTIVES (Reward Hacking): "
                f"The agent is improving its reward (delta: +{delta:.2f}) or collecting high rewards ({mean_reward:.2f}), "
                f"but Success Rate is effectively ZERO ({success_rate*100:.1f}%). "
                f"DIAGNOSIS: The agent has found a way to 'game' the system (e.g., spinning, standing near key) without solving the task. "
                f"FIX: You MUST punish the agent for not progressing, or cap the maximum reward for sub-goals."
            )

        # 2. THE "SPARSE REWARD" CHECK (Stagnation)
        # Flat delta and no success means the agent is just wandering randomly
        if abs(delta) < 0.05 and success_rate < 0.05:
            return (
                "STAGNATION DETECTED: The agent is learning NOTHING (Reward Delta near zero, Success near zero). "
                "DIAGNOSIS: The reward signal is too weak."
                "FIX: Create a DENSE reward field to guide the agent."
            )
            
        # 3. THE "UNLEARNING" CHECK (Collapse)
        if delta < -0.5:
             return (
                 f"WARNING: TRAINING COLLAPSE. The reward trend is negative (delta: {delta:.2f}). "
                 "The agent is actively getting worse. "
                 "FIX: Check if you are applying too many negative penalties (living costs) that drive the agent to suicide/early termination."
             )

        # 4. REAL PROGRESS
        if delta > 0.1 and success_rate > 0.0 and success_rate < 0.5:
            return (
            f"PROMISING: Valid Learning Detected. During training the reward is increasing (+{delta:.2f}) "
            f"But success rate ({success_rate:.2f}) is still below 0.5. "
            "Performance is still modest and can improve significantly — the agent is doing barely well (he solves the env less than half of the time) it can do way better. "
            "DIAGNOSIS: The reward signal enables learning but is not yet strong or aligned enough to reach high reliability. "
            "FIX: Increase positive incentives add or boost one-time bonuses for key subgoals, "
            "and reduce any exploitable shaping so the agent is pushed toward higher success and efficiency."
            )

        # 5. big increse modify slightly the reward function
        if delta > 0.5 and success_rate >= 0.5:
            return (
            f"GOOD PROGRESS: Strong Learning Detected. During training the reward is increasing significantly (+{delta:.2f}) "
            f"And success rate ({success_rate:.2f}) is already above 0.5. "
            "DIAGNOSIS: The reward signal is effective and well-aligned, enabling the agent to learn quickly and achieve moderate success. "
            "FIX: Consider fine-tuning the reward function to further enhance performance, "
            "such as optimizing incentives for efficiency or refining sub-goal rewards to push success rates even higher."
            )          

        return ""

    def _generate_feedback(self, metrics):
        """Generate stage-specific diagnostic feedback"""
        feedback = []
        
        if metrics['mean_reward'] > 30 and metrics['key_pickup_rate'] < 0.3:
            feedback.append(
                "REWARD EXPLOITATION: High rewards but low key pickup. "
                "Your distance reward is being exploited (e.g. agent stands near key without picking it up). "
                "REQUIRED FIX: (1) YOU MUST ADD a small step penalty, (2) Reduce distance reward magnitude."
            )

        # --- 1. General Diagnosis over Train and Eval
        trend_msg = self._analyze_general_trend(
            metrics.get('train_reward_delta', 0.0), 
            metrics['success_rate'],
            metrics['mean_reward']
        )
        if trend_msg:
            feedback.append(trend_msg)
            # If it's reward hacking emphasize it heavily
            if "MISALIGNED" in trend_msg:
                feedback.append("IMMEDIATE FIX REQUIRED: ")
                feedback.append("Do not focus on specific stages below until the Misalignment is fixed.")
        
        # --- 2. Stage-Specific Analysis ---        
        # KEY PICKUP (Stage 1)
        if metrics['key_pickup_rate'] < 0.2:
            feedback.append(
                "STAGE 1 FAILURE (Key): The agent almost never picks up the key. The key_pickup_rate < 0.2. "
                "CONSIDER THE FOLLOWING: "
                "(1) Ensure your distance reward guides the agent explicitly to the key location "
                "(2) Provides a massive ONE-TIME bonus for the 'pickup' action."
            )        
        # elif because if the first stage is failed do not check the next ones

        # DOOR OPEN (Stage 2)
        elif metrics['key_pickup_rate'] > 0.5 and metrics['door_open_rate'] < 0.2:
            feedback.append(
                "STAGE 2 FAILURE (Door): The agent has the key but fails to open the door. The door_open_rate < 0.2. "
                "CONSIDER THE FOLLOWING: "
                "(1) The agent likely doesn't know it needs to walk TO the door. "
                "Add a dense distance reward based on dist-to-door ONLY IF carrying_key is True."
                "(2) Provide a substantial ONE-TIME bonus for the 'open door' action."
                "(3) Check again the INTERACTION RULES"
            )
            
        # GOAL (Stage 3)
        elif metrics['door_open_rate'] > 0.5 and metrics['success_rate'] < 0.2:
            feedback.append(
                "STAGE 3 FAILURE (Goal): The agent opens the door but wanders off. The success_rate < 0.2. "
                "CONSIDER THE FOLLOWING: "
                "Ensure that AFTER the door is open, the distance reward switches target to the GOAL."
            )

        # --- 3. Efficiency ---        
        if metrics['success_rate'] > 0.8 and metrics['mean_steps'] > 80:
            feedback.append("OPTIMIZATION: High success rate but takes too many steps (mean_steps > 80) "
                            "CONSIDER THE FOLLOWING: "
                            "Try adding a small step penalty to encourage speed.")

        # Final Formatting
        if not feedback:
            return "Observation: Performance is average. No specific critical errors detected. Try to tune it to perform better."
            
        print("\nGenerated Feedback")
        print(feedback)
        return "\n\n".join(feedback)
    

    def find_best_RwdFunc(self):
        #prompt che serve per inviare inieme a old code per feedback
        context_prompt = MINIGRID_API_CONTEXT_DOORKEY.format(
            width=self.env_width,
            height=self.env_height,
        )

        # prompt iniziale senza feedback che serve per la prima generazione
        current_prompt = EUREKA_INITIAL_PROMPT_DOORKEY.format(
            width=self.env_width,
            height=self.env_height,
        )
            
        for i in range(self.reflection_iterations):
            print(f"\n>>> Iteration to improve HEURISTIC: {i+1}/{self.reflection_iterations}")
            
            # 1. Generate
            # no need to pass through the cache in this approach so call irectly llm_model._get_raw_response
            response = self.llm_model._get_raw_response(current_prompt, 
                                                        False # it is the generate_explanation parameter, not used by DeepSeek but it must be passed
                                                        )
            if response is None:
                print(f"[Eureka Search] iteration:{i}/{self.reflection_iterations} --> No response from LLM, skipping evaluation.")
                continue

            code = self.clean_code_string(response)            
            if code == "":
                print(f"[Eureka Search] iteration:{i}/{self.reflection_iterations} --> No valid code generated, skipping evaluation.")
                print("The ERROR IS IN THE LLM clean_code_string FUNCTION")
                print(f"LLM Response was:\n{response}")
                continue
                
            #----evaluate v2
            eval_stats, err = self.evaluate_candidate_v2(code)
            print("- "*20 + "\nEvaluation Metrics:")
            for k, v in eval_stats.items():
                print(f"  {k}: {v}")
            print("- "*20)

            mean_rew = eval_stats['mean_reward'] # the mean of the total_summed_reward per episode
            success = eval_stats['success_rate']

            # 3. Update Best just to check
            if mean_rew > self.best_reward:
                self.best_reward = mean_rew
                self.best_code = code
                
                if "deepseek-r1" in self.llm_model.model_name or "deepseek-r1:8b" in self.llm_model.model_name:
                    model_name = "DeepSeekR1_8b"
                elif "deepseek-v3.1" in self.llm_model.model_name or "deepseek-v3" in self.llm_model.model_name or "671b-cloud" in self.llm_model.model_name:
                    model_name = "DeepSeek671b"
                elif "phi3.5" in self.llm_model.model_name or "phi3_5" in self.llm_model.model_name or self.llm_model.model_name.startswith("phi3"):
                    model_name = "Phi3_5"
                elif "sonar-reasoning-pro" in self.llm_model.model_name.lower() or "sonar" in self.llm_model.model_name.lower():
                    model_name = "Perplexity"
                else:
                    model_name = "UnknownModel"

                if "DoorKey" in self.env_id:
                    name_env = "DoorKey"+str(self.env_width)+"x"+str(self.env_height)
                elif "Empty" in self.env_id:
                    name_env = "Empty"+str(self.env_width)+"x"+str(self.env_height)
                else:
                    name_env = "UnknownEnv"

                with open(f"BestRwdFunc_{name_env}_{model_name}.py", "w") as f:
                    f.write(code)

            # 4. Feedback V2 VERSION

            if err:
                # --- CRITICAL FIX ---
                # If the code crashed, the stats (0.0, -inf) are fake. 
                # Do NOT generate strategy feedback. Force LLM to focus ONLY on the bug.
                feedback_text = (
                    "CRITICAL ERROR: YOUR CODE CRASHED WITH A PYTHON ERROR! "
                    "DO NOT try to improve the reward logic.\n"
                    "DO NOT add new features.\n"
                    "DO NOT change anything except fixing the error.\n\n"
                    "You must focus EXCLUSIVELY on fixing the Python error listed in the ERROR LOGS below."
                )
            else:                
                # create a dettailed feedback based on the eval stats to add to the feedback prompt
                # Only analyze performance if the code actually ran
                feedback_text = self._generate_feedback(eval_stats)

            previous_code = code
            #if pyhton badge already present do not add again
            if "```python" not in previous_code:
                previous_code = f"\n```python\n{code}\n```"
            else:
                previous_code = f"\n{code}\n"

            # Ensure err is a string for the prompt, defaulting to "None" if it is None
            error_string = str(err) if err else "None"
            if err not in [None, "None"]:
                print("- - - - - "*10)
                print(f"[Eureka Search Error] Evaluation failed with exception")
                print(error_string)
                print("- - - - - "*10)

            feedback_prompt_body = EUREKA_FEEDBACK_PROMPT_TEMPLATE_DOORKEY.format(
                #1. the previous code submitted must be the first thing in the prompt 
                previous_code=previous_code, 
                
                # 2. only after the previous code add the part with metrics analysis
                success_rate=eval_stats['success_rate'] * 100,
                key_pickup_rate=eval_stats['key_pickup_rate'] * 100,  
                door_open_rate=eval_stats['door_open_rate'] * 100,    
                mean_steps=eval_stats['mean_steps'],                  
                mean_reward=eval_stats['mean_reward'],

                #3. error log if any
                error_log=error_string,
                feedback_text=feedback_text
            )

            #in previous version the new prompt after the initial was only the feedback prompt body
            # giving that the interaction with the LLM is stateless (no chat conversation) we need to always add the context!
            current_prompt = context_prompt + "\n\n" + feedback_prompt_body


            #[debug]
            # print("- "*20)
            # print("New Feedback prompt for LLM")
            # print(current_prompt)
            # print("-"*20)


        #=============
        # AFTER EVALUATION RETURN CODE TO DO THE FINAL BIG RUN OVER THE BEST REWARD FUNCTION
        #=============        
        print("\n\n===[EUREKA SEARCH COMPLETE]===")
        if self.best_code is None:
            print("No valid reward function was generated during the search")
            return
        else:
            print("Best Reward Function Found")
            return self.best_code
        
    
    def train_final_model(self, 
                          reward_code_str=None, 
                          final_train_epochs=200,
                          final_train_max_steps=300
                          ):
        """
        Takes the best reward function found during the search
        and trains a fresh agent for a longer duration.
        """
       
        if reward_code_str is None:
            print("[Warning] No reward_code_str provided, using the best code found during search.")
            reward_code_str = self.best_code
        else:
            if reward_code_str != self.best_code:
                print("[Warning] The provided reward_code_str does not match the best code found during search.")
            else:
                print("[Info] Using the provided reward_code_str for final training.")
    
        print("\n" + "="*60)
        print(f"STARTING FINAL TRAINING RUN\n")
        print(f'Epochs: {final_train_epochs}\nMax Steps: {final_train_max_steps}\nUsing Baseline RL: {self.pure_rl_baseline}')
        print("="*60)

        # 1. Create Environment using the BEST reward function
        # We assume make_minigrid_env can accept 'eureka_reward_code'
        final_env = make_minigrid_env(
            env_id=self.env_id,
            render_mode="rgb_array", 
            eureka_reward_code=reward_code_str, 
            use_llm_rewards=False, 
            max_steps=final_train_max_steps  
        )()

        # 2. Setup PPO with more epochs than the search phase
        final_policy = PPO(
            env=final_env,
            gamma=0.99,
            epsilon=0.2,
            epochs=final_train_epochs, 
            model_name=f"{self.pure_rl_baseline}_FINAL_BEST",

            save_pkl_model=True,  # NOW Save the final model
            track_stats=True  # Track detailed stats for final model
        )

        final_policy.batch_size = 2048  # 4096 for 8x8 / 2048 # for 5x5
        final_policy.rollout.iterations = 4096  # for 8x8 16384 / # for 5x5 4096

        # 3. Train
        final_policy.trainer(
            early_stopping_threshold=None  # No early stopping for final training
        )
        
        return final_policy


            