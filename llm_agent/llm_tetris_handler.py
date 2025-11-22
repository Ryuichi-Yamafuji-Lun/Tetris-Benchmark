# llm_tetris_handler.py

import os
import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# --- Path Setup ---
# Add the project root to path to ensure all modules and configs are found
# This assumes the script is run from the project root OR the llm_agent directory is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Core Components ---
# NOTE: The BaseAgent class (the parent) is assumed to be in base_agent.py
from llm_agent.base_agent import BaseAgent 
from llm_agent.modules.core_module import Observation
from llm_agent.modules.perception_module import PerceptionModule
from llm_agent.modules.memory_module import MemoryModule
from llm_agent.modules.reasoning_module import ReasoningModule

# --- Custom Subclasses (Must be defined for initialization) ---
# NOTE: These must exist as separate files/classes in your modules/ folder
class TetrisPerceptionModule(PerceptionModule):
    # This class exists solely to be passed to the BaseAgent's custom_modules dictionary
    pass

class TetrisReasoningModule(ReasoningModule):
    pass
# --- End Subclasses ---


# --- HELPER: Abstraction Logic (NumPy Feature Vector to Text Prompt) ---

def convert_features_to_prompt(obs_vector: np.ndarray, action_mask: np.ndarray, action_map: Dict[str, int], info: dict, tetromino_map: Dict[int, str]) -> str:
    """
    Translates the numerical feature vector (obs) and action mask into a text prompt.
    This replaces the Perception Module's job for the V1 benchmark.
    """
    features = obs_vector.flatten()
    
    if len(features) > 11:
        piece_id = int(features[11])
    else:
        piece_id = -1
    
    current_piece_name = tetromino_map.get(piece_id, "UNKNOWN")
    
    # 1. Feature Decoding: Assume standard 10 heights, 3 aggregate stats.
    feature_summary = (
        f"Column Heights Vector (C0 to C9): {features[:10].round(2).tolist()}\n"
        f"Active Piece: {current_piece_name}\n" 
        f"Aggregate Stats (Holes, Roughness, Next Piece Type, etc.): {features[10:].round(2).tolist()}"
    )
    
    # 2. Decode mask to list available actions (using the constrained map)
    # We only list actions that are both valid by the environment (mask=1) AND in our map (constrained to 5).
    valid_actions = [
        name for name, index in action_map.items() 
        if index < len(action_mask) and action_mask[index] == 1.0 and index < 5
    ]
    
    # 3. Final Prompt Assembly (Used by the Reasoning Module prompt template)
    return (
        f"Board Feature Summary: {feature_summary}\n"
        f"Available Actions: {', '.join(valid_actions)}\n"
        "Please provide your reasoning and output one action string from the list above."
    )

# --- 2. THE CONCRETE AGENT CLASS ---

class TetrisAgentHandler(BaseAgent):
    """
    The concrete orchestrator class specialized for Tetris.
    Inherits complex P-M-R logic from BaseAgent and handles config loading.
    """
    
    def __init__(self, **kwargs):
        # Configuration file paths (relative to the base_agent.py execution context)
        config_file_path = str(Path(__file__).parent / "configs" / "module_prompts.json")
        env_config_file_path = str(Path(__file__).parent / "configs" / "game_env_config.json")
        
        # Load Action Map from external file for constraint enforcement
        self.ACTION_MAP = self._load_action_map(env_config_file_path)
        
        self.TETROMINO_MAP = self._load_tetromino_map(env_config_file_path)
        
        # Call parent's constructor to set up modules and harness
        super().__init__(
            game_name="tetris",
            model_name="gemini-2.5-flash", # Now set to the free tier model
            harness=True, # Enforce full harness mode
            config_path=config_file_path, # Pass path to module_prompts.json
            custom_modules={
                "perception_module": TetrisPerceptionModule,
                "memory_module": MemoryModule, 
                "reasoning_module": TetrisReasoningModule,
            },
            observation_mode="text", # Enforce text abstraction
            **kwargs
        )
    
    def _load_action_map(self, env_config_path: str) -> Dict[str, int]:
        """Loads and processes the action mapping from the game_env_config.json file."""
        try:
            with open(env_config_path, 'r') as f:
                config = json.load(f)
            
            # The JSON contains the original 7 actions. We filter to the required 5 actions (0-4).
            full_map = config.get('action_mapping', {})
            constrained_map = {}
            for name, index in full_map.items():
                # Only include actions 0 through 4
                if index <= 4:
                    constrained_map[name.upper().replace('_', '_')] = index
            
            # Add robust parsing strings (e.g., 'left' -> 1)
            constrained_map['LEFT'] = constrained_map.get('LEFT', 1)
            constrained_map['RIGHT'] = constrained_map.get('RIGHT', 2)
            constrained_map['ROTATE_LEFT'] = constrained_map.get('ROTATE_LEFT', 3)
            constrained_map['ROTATE_RIGHT'] = constrained_map.get('ROTATE_RIGHT', 4)
            
            return constrained_map
            
        except Exception as e:
            print(f"CRITICAL: Failed to load or parse action map from {env_config_path}. Using hardcoded default.")
            # Fallback to safe 5-action map if file fails
            return {"NO_OP": 0, "LEFT": 1, "RIGHT": 2, "ROTATE_CCW": 3, "ROTATE_CW": 4}

    def _load_tetromino_map(self, env_config_path: str) -> Dict[int, str]:
        """Loads and processes the Tetromino name mapping."""
        try:
            with open(env_config_path, 'r') as f:
                config = json.load(f)
            
            # The keys are strings in the JSON ("0", "1", etc.), convert them to integers
            raw_map = config.get('tetromino_mapping', {})
            return {int(k): v for k, v in raw_map.items()}
            
        except Exception as e:
            print(f"CRITICAL: Failed to load or parse Tetromino map from {env_config_path}. Using hardcoded default.")
            # Fallback to confirmed Tetromino names/IDs if file fails
            return {0: "I_Line", 1: "O_Square", 2: "T_Piece", 3: "S_Piece", 4: "Z_Piece", 5: "J_Piece", 6: "L_Piece"}


    # NOTE: The BaseAgent's get_action logic is INHERITED and automatically executes the harness.
    # We define the final entry point for the benchmark runner:
    
# --- 3. THE FINAL BENCHMARK ENTRY POINT (Callable by test.py) ---

AGENT_INSTANCE = TetrisAgentHandler() 

def agent_action(obs: np.ndarray, info: dict) -> int:
    """
    The final function called by the test.py runner. It executes the full LLM policy.
    """
    # 1. Abstraction: Convert numerical features and mask to the text prompt
    # UPDATED CALL SITE: Pass info and the TETROMINO_MAP
    prompt_text = convert_features_to_prompt(
        obs, 
        info["action_mask"], 
        AGENT_INSTANCE.ACTION_MAP,
        info, 
        AGENT_INSTANCE.TETROMINO_MAP
    )
    
    # 2. Package into Observation object
    llm_observation = Observation(textual_representation=prompt_text)
    
    # 3. Run Harness (inherited from BaseAgent)
    # The inherited get_action method runs P->M->R and returns (action_plan, processed_obs)
    action_plan, _ = AGENT_INSTANCE.get_action(llm_observation) 
    
    # 4. Mapping: Convert string action to integer
    action_str = action_plan.get("action", "NO_OP")
    
    # Normalize and map the string action to the integer index
    final_action_str = action_str.upper().strip().replace(' ', '_')
    
    print("finactionstr:", final_action_str)
    # Returns the integer action index, defaulting to 0 (NO_OP) if not found.
    action = AGENT_INSTANCE.ACTION_MAP.get(final_action_str, 0)

    print("return action:", action)

    return action