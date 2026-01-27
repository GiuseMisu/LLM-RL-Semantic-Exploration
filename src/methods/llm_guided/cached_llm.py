import os
import json
import statistics
import ast
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.methods.llm_guided.llm_shared_utils import BaseLLMClient

class RobustCachedLLMClient(BaseLLMClient):
    """
    A Wrapper Client that adds caching and physics-based guardrails:
    IT HANDLES BOTH THE ENVIRONMENTS
    1. Persistence: Caches rewards to disk (JSON) to ensure deterministic runs
    2. Stability: Uses Median Voting (queries LLM for N times) to ignore hallucinations
    3. Reflection Guardrails: Re-queries LLM with hints when reward seems inconsistent
    """

    def __init__(self, real_llm_client: BaseLLMClient, cache_path="llm_reward_cache.json", 
                 voting_samples=3, mode: str = None, max_reflection_attempts: int = 2):
        """
        Args:
            real_llm_client: An instance of GeminiLLMClient, Phi35LLMClient, etc
            cache_path (str): File path to store the JSON cache
            voting_samples (int): How many times to query the LLM on a cache miss (3 or 5)
            max_reflection_attempts (int): Max times to re-query LLM if reward seems wrong
        """
        
        # inherit the system prompt from the real client to pass it correctly if needed
        super().__init__(system_prompt=real_llm_client.system_prompt)
        
        self.client = real_llm_client
        self.cache_path = cache_path
        self.voting_samples = voting_samples
        self.max_reflection_attempts = max_reflection_attempts
        
        # Load existing cache or create fresh
        self.cache = self._load_cache()

        # --- DETECT ENV ---
        # check the system prompt to decide which guardrails to apply
        if mode is not None:
            self.mode = mode.upper()
            if "empty" in mode.lower() or "minigrid-empty" in mode.lower() :
                self.mode = "EMPTY"
            elif "door" in mode.lower()  and "key" in mode.lower()  or "doorkey" in mode.lower() :
                self.mode = "DOORKEY"
        else:
            prompt = (self.client.system_prompt or "").lower()
            if "empty" in prompt or "minigrid-empty" in prompt:
                self.mode = "EMPTY"
            elif "door" in prompt and "key" in prompt or "doorkey" in prompt:
                self.mode = "DOORKEY"
            else:
                raise Exception("[cached_llm.py] Unrecognized ENVIRONMENT FROM PROMPT ")

        print(f"[RobustCachedLLMClient] Mode: {self.mode} / Cache: {self.cache_path}")
        # Tracking statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "reflection_attempts": 0,
            "reflection_successes": 0  # When reflection fixed the reward
        }

    def _load_cache(self):
        #Loads the JSON cache from disk
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    print(f"[CACHE] Loaded existing cache from {self.cache_path}")
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"[CACHE] Warning: Cache file corrupted. Starting fresh.")
                return {}
        return {}

    def _save_cache(self):
        #Saves the current cache to disk        
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"[CACHE] Error saving cache: {e}")


    def _build_DOORKEY_reflection_hint(self, obs_data: dict, proposed_reward: float) -> str:
        """
        MINIMAL Reflection Guardrail - Only catches critical errors.
        Triggers ONLY when the reward is drastically wrong and could damage training.
        
        Critical cases:
        1. Goal is HERE or REACHABLE but reward is low (missing the win/near-win)
        2. Agent has nothing actionable but reward is very high (false positive)
        3. Agent has Key + Door is OPEN but reward is low (missing phase 2 success)
        """

        if not (-0.1 <= proposed_reward <= 1):
            return (
                "\n\n[CRITICAL REFLECTION]\n"
                f"CURRENT STATE: The proposed reward ({proposed_reward}) is outside valid range [-0.1, 1].\n"
                "SUGGESTION: Please provide a reward within the valid range according to the SCORING GUIDELINES."
            )
        
        key_info = obs_data.get('Key', '')
        door_info = obs_data.get('Door', '')
        goal_info = obs_data.get('Goal', '')
        
        # Parse critical state flags
        key_held = "In Inventory" in key_info
        key_is_reachable = "<REACHABLE>" in key_info
        door_is_reachable = "<REACHABLE>" in door_info
        door_is_open = "state=Open" in door_info or "state=Unlocked" in door_info
        goal_is_reachable = "<REACHABLE>" in goal_info
        goal_is_here = "dist=0" in goal_info or "dir=Here" in goal_info
        
        # =====================================================================
        # CASE 1: FALSE NEGATIVES (GOOD STATE but low reward)
        # =====================================================================
        if goal_is_here and proposed_reward < 0.8:
            return (
                "\n\n[CRITICAL REFLECTION]\n"
                "CURRENT STATE: The agent is standing ON the Goal (dist=0 / dir=Here).\n"
                "SUGGESTION: This is the WIN condition.\n"
                "Provide the correct reward that is aligned with the SCORING GUIDELINES"
            )
        
        if (goal_is_reachable or "dist=1" in goal_info) and proposed_reward < 0.6:
            return (
                "\n\n[CRITICAL REFLECTION]\n"
                "CURRENT STATE: The Goal is very close (dist=1) or marked <REACHABLE>.\n"
                "SUGGESTION: Being adjacent to the goal should receive higher reward.\n"
                "Provide the correct reward that is aligned with the SCORING GUIDELINES"
            )
        
        if door_is_open and proposed_reward < 0.5:
             return (
                "\n\n[CRITICAL REFLECTION]\n"
                "CURRENT STATE: The Door is OPEN\n"
                "SUGGESTION: This is the last step of phase 2, the path to the Goal is now clear. The door is open, so it does not matter if you held the key or not.\n"
                "Provide the correct reward that is aligned with the SCORING GUIDELINES"
            )
        
        # =====================================================================
        # CASE 2: FALSE POSITIVES (High reward but state doesn't justify it)
        # distinguish between Phase 1, Phase 2, and Phase 3 limits.
        # =====================================================================
        if proposed_reward >= 0.7:
            
            # Sub-case A: High reward, but Agent doesn't even have the Key yet
            if not key_held and not key_is_reachable:
                # If the door is already OPEN, not having the key is VALID for a high reward.
                # We must return "" so the loop accepts the high reward.
                if door_is_open:
                    return ""
                else:
                    return (
                        "\n\n[CRITICAL REFLECTION]\n"
                        "CURRENT STATE: The Agent does NOT possess the Key yet.\n"
                        "SUGGESTION: High reward is impossible in Phase 1 (Finding the Key). So Low reward\n"
                        "Provide the correct reward that is aligned with the SCORING GUIDELINES"
                    )

            # Sub-case B: High reward, Key is held, but Door is still Locked/Closed and far
            if key_held and not door_is_open and not door_is_reachable:
                return (
                    f"\n\n[CRITICAL REFLECTION]\n"
                    f"CURRENT STATE: The Door is currently Locked/Closed and NOT reachable.\n"
                    "SUGGESTION: Merely holding the key while wandering implies a moderate reward (reward should be: not too high and not too low).\n"
                    "Provide the correct reward that is aligned with the SCORING GUIDELINES"                    
                )    
  
        # No critical issues detected
        return ""
    
    def _DOORKEY_guardrails(self, observation_str: str, proposed_reward: float,
                            verbose: bool = False, generate_explanation: bool = False) -> float:
        """
        Reflection-based guardrail for DoorKey environment.
        If the reward seems inconsistent, re-queries LLM with hints.
        Will attempt up to max_reflection_attempts times.
        NB: The guardrails take in input the median of 3 llm query on the same state 
            So it judges the reward based on that median value. if it is wrong, it tries to fix it.
            and queries again the LLM --BUT THIS TIME IS A SINGLE QUERY-- with the reflection hint added to the observation.
            --> WITH REFLECTION SINGLE QUERY NO MEDIAN OF THREE WITH ALSO THE REFLECTION HINT.
        """
        try:
            # Parse the observation string back into a Python Dictionary
            # The textualizer uses single quotes, so json.loads might fail. 
            # ast.literal_eval is safer for Python-dictionary-like strings.
            obs_data = ast.literal_eval(observation_str)
        except Exception as e:
            # If parsing fails, we cannot verify. Return original reward to be safe
            print(f"[DOORKEY-GUARDRAIL] Warning: Failed to parse observation: {e}")
            return proposed_reward

        current_reward = proposed_reward
        
        for attempt in range(self.max_reflection_attempts):
            # Build reflection hint based on detected issues
            reflection_hint = self._build_DOORKEY_reflection_hint(obs_data, current_reward)
            
            # If no issues detected, reward is acceptable
            if reflection_hint == "":
                #if attempt > 0 and verbose:
                if attempt > 0: #always print for [debug]
                    print(f"[GUARDRAIL] Reflection successful in {attempt} attempts => Proposed Reward: {proposed_reward} | Final Reward: {current_reward}")
                    self.stats["reflection_successes"] += 1
                return current_reward
            
            # Issue detected -> Query LLM again with reflection
            self.stats["reflection_attempts"] += 1
            
            if verbose:
                print(f"[GUARDRAIL] Attempt {attempt + 1}/{self.max_reflection_attempts}: "
                      f"Reward {current_reward} seems inconsistent. Requesting reflection...")
            
            # The reflection prompt is added to the OBSERVATION PROMPT (CURRENT STATER TEXTUAL DESCRIPTION)
            reflection_prompt = observation_str + reflection_hint
            
            # Query LLM with reflection
            reflected_reward = self.client.get_reward(
                observation=reflection_prompt,  # now the observation may have the reflection included
                verbose=verbose, 
                generate_explanation=generate_explanation
            )
            
            if verbose:
                print(f"[GUARDRAIL] Attempt {attempt + 1}: {current_reward} -> {reflected_reward}")
            
            current_reward = reflected_reward
        
        # Max attempts reached - return whatever we have
        if verbose:
            print(f"[GUARDRAIL] Max reflection attempts ({self.max_reflection_attempts}) reached. "
                  f"Final reward: {current_reward}")
        
        return current_reward
    
    def _EMPTY_guardrails(self, observation_str: str, proposed_reward: float,
                          verbose: bool = False, generate_explanation: bool = False) -> float:
        # No reflection implemented for EMPTY env
        return proposed_reward

    def _get_raw_response(self, prompt: str, generate_explanation: bool) -> str:
        #Required implementation of abstract method.
        #Since this is a wrapper, delegate to the real client.
        return self.client._get_raw_response(prompt, generate_explanation)

    def try_cached_get_reward(self, observation: str, verbose: bool = False, generate_explanation: bool = False) -> float:
        """The main public method called by the RL agent"""
        
        # 1. Normalize the Key
        cache_key = observation.strip()

        # 2. Check Cache (Hit)
        if cache_key in self.cache:
            self.stats["hits"] += 1
            if verbose:
                print(f"[CACHE HIT] Reward: {self.cache[cache_key]}")
            return float(self.cache[cache_key])

        # 3. Cache Miss
        self.stats["misses"] += 1
        if verbose:
            print(f"[CACHE MISS] querying LLM {self.voting_samples} times")

        rewards = []
        
        # --- VOTING LOOP ---
        for _ in range(self.voting_samples):
            r = self.client.get_reward(cache_key, verbose=verbose, generate_explanation=generate_explanation)
            rewards.append(r)
        
        # 4. Calculate Median
        final_reward = statistics.median(rewards)
        if verbose:
            print(f"    Raw Votes: {sorted(rewards)} -> Median: {final_reward}")

        # 5. Apply Reflection-Based Guardrails
        if self.mode == "DOORKEY":
            guarded_reward = self._DOORKEY_guardrails(
                cache_key, final_reward, verbose, generate_explanation
            )
        else:  # EMPTY mode
            guarded_reward = self._EMPTY_guardrails(
                cache_key, final_reward, verbose, generate_explanation
            )
        
        if guarded_reward != final_reward and verbose:
            print(f"   -> [GUARDRAIL] Final correction: {final_reward} => {guarded_reward}\n")

        # 6. Save to Cache
        self.cache[cache_key] = guarded_reward
        self._save_cache()

        return guarded_reward

    def print_stats_summary(self) -> str:
        """Returns a formatted summary of guardrail statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]

        summary = (
            f"\n{'='*50}\n"
            f"CACHE & GUARDRAIL STATISTICS\n"
            f"{'='*50}\n"
            f"Total Requests:       {total_requests}\n"
            f"Cache Hits:           {self.stats['hits']} ({100*self.stats['hits']/max(1,total_requests):.1f}%)\n"
            f"Cache Misses:         {self.stats['misses']}\n"
            f"Reflection Attempts:  {self.stats['reflection_attempts']}\n"
            f"Reflection Successes: {self.stats['reflection_successes']}\n"
            f"{'='*50}"
        )
        print(summary)


if __name__ == "__main__":
    #=======================
    # PHI 3.5 TESTS
    #=======================
    from src.methods.llm_guided.phi3_5 import Phi35LLMClient
    from src.methods.llm_guided.llm_shared_utils import DOOR_KEY_SYSTEM_PROMPT

    # 1. Initialize the Real Client
    try:        
        real_client = Phi35LLMClient(system_prompt=DOOR_KEY_SYSTEM_PROMPT)
        print(f"Client Initialized Model: {real_client.model_name}")
    except Exception as e:
        print(e)
        sys.exit(1)
        
    # 2. Wrap it to get the cached and guardrail version
    # Use a fresh cache for testing
    cache_name = "test_guardrail_cache.json"
    # Delete old cache to force fresh LLM queries
    if os.path.exists(cache_name):
        os.remove(cache_name)
        print(f"[TEST] Removed old cache: {cache_name}")
    
    cached_client = RobustCachedLLMClient(
        real_client, 
        cache_path=cache_name, 
        voting_samples=3,  
        mode="DOORKEY",
        max_reflection_attempts=2
    )
    
    print("\n" + "="*70)
    print("GUARDRAIL CRITICAL CASE TESTS (Full LLM Integration)")
    print("="*70)
    
    # --- CRITICAL CASE 1A: Goal is HERE ---
    print("\n--- CASE 1A: Agent ON the Goal (dist=0) ---")
    print("Expected reward: 1.0 (Phase 3 win condition)")
    obs_goal_here = (
        "{ 'Agent': { 'pos': (3, 3), 'facing': 'South', 'inventory': 'Yellow Key' }, "
        "'Key': 'In Inventory (Carried)', "
        "'Door': 'loc=(2, 2), dist=2, dir=Behind, state=Open', "
        "'Goal': 'loc=(3, 3), dist=0, dir=Here' }"
    )
    result_1a = cached_client.try_cached_get_reward(obs_goal_here, verbose=True, generate_explanation=False)
    print(f">>> FINAL REWARD: {result_1a}\n")
    
    # --- CRITICAL CASE 1B: Goal is REACHABLE ---
    print("\n--- CASE 1B: Goal REACHABLE (dist=1, dir=Front) ---")
    print("Expected reward: 0.9 (Phase 3 one step from win)")
    obs_goal_reachable = (
        "{ 'Agent': { 'pos': (3, 2), 'facing': 'South', 'inventory': 'Yellow Key' }, "
        "'Key': 'In Inventory (Carried)', "
        "'Door': 'loc=(2, 2), dist=1, dir=Left, state=Open', "
        "'Goal': 'loc=(3, 3), dist=1, dir=Front <REACHABLE>' }"
    )
    result_1b = cached_client.try_cached_get_reward(obs_goal_reachable, verbose=True, generate_explanation=False)
    print(f">>> FINAL REWARD: {result_1b}\n")
    
    # --- CRITICAL CASE 2: Wandering with high reward (hallucination) ---
    # This tests if LLM incorrectly gives high reward when nothing is actionable
    print("\n--- CASE 2: Wandering (nothing actionable) ---")
    print("Expected reward: 0.1 (wandering with key, nothing reachable)")
    obs_wandering = (
        "{ 'Agent': { 'pos': (1, 1), 'facing': 'North', 'inventory': 'Yellow Key' }, "
        "'Key': 'In Inventory (Carried)', "
        "'Door': 'loc=(4, 4), dist=6, dir=Behind, state=Locked', "
        "'Goal': 'loc=(5, 5), dist=8, dir=Behind' }"
    )
    result_2 = cached_client.try_cached_get_reward(obs_wandering, verbose=True, generate_explanation=False)
    print(f">>> FINAL REWARD: {result_2}\n")
    
    # --- CRITICAL CASE 3: Door OPEN + Key held ---
    print("\n--- CASE 3: Door OPEN + Key Held (Phase 2 success) ---")
    print("Expected reward: 0.7 (door unlocked, path to goal clear)")
    obs_door_open = (
        "{ 'Agent': { 'pos': (2, 3), 'facing': 'East', 'inventory': 'Yellow Key' }, "
        "'Key': 'In Inventory (Carried)', "
        "'Door': 'loc=(3, 3), dist=1, dir=Front <REACHABLE>, state=Open', "
        "'Goal': 'loc=(5, 5), dist=4, dir=Right' }"
    )
    result_3 = cached_client.try_cached_get_reward(obs_door_open, verbose=True, generate_explanation=False)
    print(f">>> FINAL REWARD: {result_3}\n")
    
    # --- VALID CASE: Key Reachable (should NOT trigger guardrail) ---
    print("\n--- VALID CASE: Key Reachable (no guardrail expected) ---")
    print("Expected reward: 0.5 (key is reachable)")
    obs_key_reachable = (
        "{ 'Agent': { 'pos': (2, 2), 'facing': 'East', 'inventory': 'None' }, "
        "'Key': 'loc=(3, 2), dist=1, dir=Front <REACHABLE>', "
        "'Door': 'loc=(4, 4), dist=4, dir=Right, state=Locked', "
        "'Goal': 'loc=(5, 5), dist=6, dir=Right' }"
    )
    result_valid = cached_client.try_cached_get_reward(obs_key_reachable, verbose=True, generate_explanation=False)
    print(f">>> FINAL REWARD: {result_valid}\n")
    
    # --- EDGE CASE: Phase 1 wandering (no key visible) ---
    print("\n--- EDGE CASE: Wandering without Key ---")
    print("Expected reward: 0.1 (wandering, key not found)")
    obs_no_key = (
        "{ 'Agent': { 'pos': (1, 1), 'facing': 'West', 'inventory': 'None' }, "
        "'Key': 'Not Found', "
        "'Door': 'loc=(4, 4), dist=6, dir=Behind, state=Locked', "
        "'Goal': 'Unknown' }"
    )
    result_edge = cached_client.try_cached_get_reward(obs_no_key, verbose=True, generate_explanation=False)
    print(f">>> FINAL REWARD: {result_edge}\n")

    cached_client.print_stats_summary()


    
    # =========================================================================
    # GUARDRAIL TEST CASES - Testing the 3 Critical Cases
    # =========================================================================
    
    print("\n" + "="*70)
    print("CHECK IF THE GUARDRAIL UNDERSTADN WHEN TO ACTION THE REFLECTION GUARDRAIL (No caching, no voting)")
    print("="*70)
    
    # --- CRITICAL CASE 1A: Goal is HERE but reward is low ---
    print("\n--- CASE 1A: Goal HERE + Low Reward (Should trigger reflection) ---")
    obs_goal_here_low = "{ 'Agent': { 'pos': (3, 3), 'facing': 'South', 'inventory': 'Yellow Key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(2, 2), dist=2, dir=Behind, state=Open', 'Goal': 'loc=(3, 3), dist=0, dir=Here' }"
    # Simulate LLM giving wrong reward
    test_reward_1a = cached_client._build_DOORKEY_reflection_hint(
        ast.literal_eval(obs_goal_here_low), 
        proposed_reward=0.3  # Way too low for being ON the goal
    )
    print(f"Proposed: 0.3 | Hint triggered: {bool(test_reward_1a)}")
    if test_reward_1a:
        print(test_reward_1a)
    
    # --- CRITICAL CASE 1B: Goal is REACHABLE but reward is low ---
    print("\n--- CASE 1B: Goal REACHABLE + Low Reward (Should trigger reflection) ---")
    obs_goal_reachable_low = "{ 'Agent': { 'pos': (3, 2), 'facing': 'South', 'inventory': 'Yellow Key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(2, 2), dist=1, dir=Left, state=Open', 'Goal': 'loc=(3, 3), dist=1, dir=Front <REACHABLE>' }"
    test_reward_1b = cached_client._build_DOORKEY_reflection_hint(
        ast.literal_eval(obs_goal_reachable_low), 
        proposed_reward=0.2  # Way too low for goal being reachable
    )
    print(f"Proposed: 0.2 | Hint triggered: {bool(test_reward_1b)}")
    if test_reward_1b:
        print(test_reward_1b)
    
    # --- CRITICAL CASE 2: High reward but nothing actionable ---
    print("\n--- CASE 2: High Reward + Nothing Actionable (Should trigger reflection) ---")
    obs_wandering_high = "{ 'Agent': { 'pos': (1, 1), 'facing': 'North', 'inventory': 'Yellow Key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(4, 4), dist=6, dir=Behind, state=Locked', 'Goal': 'loc=(5, 5), dist=8, dir=Behind' }"
    test_reward_2 = cached_client._build_DOORKEY_reflection_hint(
        ast.literal_eval(obs_wandering_high), 
        proposed_reward=0.8  # Way too high for wandering
    )
    print(f"Proposed: 0.8 | Hint triggered: {bool(test_reward_2)}")
    if test_reward_2:
        print(test_reward_2)
    
    # --- CRITICAL CASE 3: Key held + Door OPEN but low reward ---
    print("\n--- CASE 3: Key Held + Door OPEN + Low Reward (Should trigger reflection) ---")
    obs_door_open_low = "{ 'Agent': { 'pos': (2, 3), 'facing': 'East', 'inventory': 'Yellow Key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(3, 3), dist=1, dir=Front <REACHABLE>, state=Open', 'Goal': 'loc=(5, 5), dist=4, dir=Right' }"
    test_reward_3 = cached_client._build_DOORKEY_reflection_hint(
        ast.literal_eval(obs_door_open_low), 
        proposed_reward=0.2  # Way too low for door being open
    )
    print(f"Proposed: 0.2 | Hint triggered: {bool(test_reward_3)}")
    if test_reward_3:
        print(test_reward_3)
    
    # --- VALID CASE: Should NOT trigger ---
    print("\n--- VALID CASE: Correct Reward (Should NOT trigger reflection) ---")
    obs_valid = "{ 'Agent': { 'pos': (2, 2), 'facing': 'East', 'inventory': 'None' }, 'Key': 'loc=(3, 2), dist=1, dir=Front <REACHABLE>', 'Door': 'loc=(4, 4), dist=4, dir=Right, state=Locked', 'Goal': 'loc=(5, 5), dist=6, dir=Right' }"
    test_reward_valid = cached_client._build_DOORKEY_reflection_hint(
        ast.literal_eval(obs_valid), 
        proposed_reward=0.5  # Correct for key reachable
    )
    print(f"Proposed: 0.5 | Hint triggered: {bool(test_reward_valid)}")
    