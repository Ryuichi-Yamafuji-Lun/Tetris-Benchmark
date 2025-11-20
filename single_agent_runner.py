import argparse
import os
import json
import datetime
import time
import numpy as np
import yaml
from typing import Any
import sys
import re
import random

sys.path.insert(1, 'Tetris-Benchmark/agents/llm_agent/')
sys.path.insert(1, 'Tetris-Benchmark/custom_04_tetris')

from tetris_gymnasium.envs.tetris import Tetris
from base_agent import BaseAgent 
from tetrisEnv import TetrisEnv
from modules import PerceptionModule, ReasoningModule # Observation is imported by Env


import gymnasium as gym

from typing import Any, Dict

game_config_mapping = {
    "tetris": "custom_04_tetris",
}

def str_to_bool(v):
    """Convert string boolean values to actual booleans for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(defaults_map=None, argv_to_parse=None):
    parser = argparse.ArgumentParser(description="Run GamingAgent for a specified Gym Environment.")
    # Game name will be set by defaults_map from prelim_parser, so not strictly required here.
    # A check after parsing will ensure it has a value.
    parser.add_argument("--game_name", type=str, default="Tetris", 
                        help="Name of the game (e.g., twenty_forty_eight, sokoban). Set by prelim parser.")
    parser.add_argument("--config_root_dir", type=str, default="/Users/rirando/Documents/tetris/Tetris-Benchmark",
                        help="Root directory for agent configurations.")
    parser.add_argument("--model_name", type=str, default="o3-mini",
                        help="Name of the model for the agent.")
    parser.add_argument("--harness", action="store_true",
                        help="Use perception-memory-reasoning pipeline (harness mode). Default is False.")
    parser.add_argument("--multiagent_arg", type=str, default="single",
                        choices=["single", "multi"], help="Multi-agent mode configuration.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of game episodes.")
    parser.add_argument("--observation_mode", type=str, default="vision",
                        choices=["vision", "text", "both"], help="Agent's observation mode.")
    parser.add_argument("--max_memory", type=int, default=20, help="Agent's max memory entries.")
    parser.add_argument("--use_reflection", type=str_to_bool, default=True, help="Enable reflection in memory module. Default is True.")
    parser.add_argument("--use_perception", type=str_to_bool, default=True, help="Enable perception API calls for image processing. Default is True.")
    parser.add_argument("--use_summary", type=str_to_bool, default=False, help="Enable trajectory summarization in memory module. Default is False.")
    parser.add_argument("--token_limit", type=int, default=100000, help="Token limit for the agent's input.")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument("--use_custom_prompt", action="store_true", help="If set, will use the custom prompt from module_prompts.json if present.")
    parser.add_argument("--scaffolding", type=str, default=None, help="Grid dimensions as '(rows,cols)' for coordinate grid on images, e.g., '(5,5)'. Default is None.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment.")
    # Env type is fixed to custom gym for this runner

    # Serving-related arguments
    parser.add_argument(
        "--modal_url",
        type=str,
        default=None,
        help="Optional URL for a Modal‑hosted inference endpoint passed to BaseAgent.",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default=None,
        help="Optional URL for a vLLM inference endpoint passed to BaseAgent.",
    )
    # First parse args with just command line values
    if argv_to_parse:
        args = parser.parse_args(argv_to_parse)
    else:
        args = parser.parse_args()

    # Store original command line and default values
    args._cli_values = {}
    for action in parser._actions:
        if action.dest != 'help':
            args._cli_values[action.dest] = getattr(args, action.dest)

    # Store YAML defaults for reference but don't apply them yet
    args._yaml_defaults = defaults_map if defaults_map else {}

    # Only apply YAML defaults for parameters that:
    # 1. Weren't explicitly set on command line
    # 2. Are using their built-in defaults
    # 3. Have a different value in YAML
    if defaults_map:
        for param_name, yaml_value in defaults_map.items():
            if yaml_value is not None:
                param_on_cli = f"--{param_name.replace('_', '-')}" in sys.argv
                if not param_on_cli:
                    # For model_name, always use CLI default
                    if param_name == "model_name":
                        continue
                    # For other parameters, use YAML if different from CLI default
                    cli_value = getattr(args, param_name)
                    if cli_value != yaml_value:
                        setattr(args, param_name, yaml_value)

    return args

def create_environment(game_name_arg: str, 
                       obs_mode_arg: str, 
                       config_dir_name_for_env_cfg: str, # For loading game_env_config.json
                       cache_dir_for_adapter: str,
                       harness: bool = False,
                       multiagent_arg: str = "single",
                       setseed: int =42):
    """Creates and returns a game environment instance based on the game name."""
    game_env_path = "/Users/rirando/Documents/tetris/Tetris-Benchmark/custom_04_tetris/"
    env_specific_config_path = os.path.join(game_env_path, "game_env_config.json")
    env_init_params = {} # Will be populated based on the specific game

    with open(env_specific_config_path, 'r') as f:
            env_specific_config = json.load(f)
            env_init_kwargs = env_specific_config.get('env_init_kwargs', {})
            env_init_params['board_width'] = env_init_kwargs.get('board_width', 10)
            env_init_params['board_height'] = env_init_kwargs.get('board_height', 20)
            env_init_params['gravity'] = env_init_kwargs.get('gravity', True)
            env_init_params['render_upscale'] = env_init_kwargs.get('render_upscale', 25)
            env_init_params['queue_size'] = env_init_kwargs.get('queue_size', 4)
            env_init_params['render_mode_for_make'] = env_specific_config.get('render_mode_for_make', 'human') # Corresponds to TetrisEnv render_mode
            env_init_params['max_stuck_steps_for_adapter'] = env_specific_config.get('max_unchanged_steps_for_termination', 30)

    print(f"Initializing environment: {game_name_arg} with params: {env_init_params}")
    env = TetrisEnv(
        render_mode=env_init_params.get('render_mode_for_make'),
        board_width=env_init_params.get('board_width'),
        board_height=env_init_params.get('board_height'),
        gravity=env_init_params.get('gravity'),
        render_upscale=env_init_params.get('render_upscale'),
        queue_size=env_init_params.get('queue_size'),
        # Adapter related params
        game_name_for_adapter=game_name_arg,
        observation_mode_for_adapter=obs_mode_arg,
        agent_cache_dir_for_adapter=cache_dir_for_adapter,
        game_specific_config_path_for_adapter=env_specific_config_path,
        max_stuck_steps_for_adapter=env_init_params.get('max_stuck_steps_for_adapter')
        # seed will be passed during reset, not __init__ for TetrisEnv as per its definition
    )
    return env

def run_game_episode(agent: BaseAgent, game_env: gym.Env, episode_id: int, args: argparse.Namespace):
    """Run a single episode of the game."""
    # Pass episode_id to env.reset
    agent_observation, last_info = game_env.reset(max_memory=args.max_memory, seed=args.seed, episode_id=episode_id)
    if args.seed is not None: args.seed += 1 # Increment seed for next potential run

    total_reward_for_episode = 0.0
    total_perf_score_for_episode = 0.0
    final_step_num = 0

    for step_num in range(args.max_steps_per_episode):
        final_step_num = step_num + 1
        game_env.render() # Call env's render method directly

        start_time = time.time()
        action_dict, processed_agent_observation = agent.get_action(agent_observation)
        end_time = time.time()
        time_taken_s = end_time - start_time

        #tetris case - in lmgame    
        action_str = None
        if action_dict and action_dict.get("action") is not None:
            action_str = action_dict.get("action")
        
        action_str_agent = "None" # Default to "None" string if no valid action
        if action_str:
            action_str_agent = str(action_str).strip().lower()
        
        thought_process = action_dict.get("thought", "") if action_dict else "No thought process due to API failure."

        # --- MODIFIED: Extract raw LLM output to pass to env.step ---
        raw_llm_output_for_env = None

        if action_dict:
            if "raw_response_str" in action_dict and isinstance(action_dict["raw_response_str"], str):
                raw_llm_output_for_env = action_dict["raw_response_str"]
        else:
            print("[Runner DEBUG] action_dict is None") # DEBUG
        
        # Conditionally pass raw_llm_output_for_next_obs
        step_args = {
            "agent_action_str": action_str_agent,
            "thought_process": thought_process,
            "time_taken_s": time_taken_s
        }
        if args.game_name == "ace_attorney":
            step_args["raw_llm_output_for_next_obs"] = raw_llm_output_for_env
        
        # Step the environment using the new signature, including agent action details
        agent_observation, reward, terminated, truncated, last_info, total_perf = game_env.step(**step_args)

        # Inherit game trajectory
        agent_observation.game_trajectory = processed_agent_observation.game_trajectory
            
        total_reward_for_episode += reward
        total_perf_score_for_episode += total_perf

        if terminated or truncated:
            break
            
    # game_env.close() is called after all runs are complete in main

    final_score_from_env = float(last_info.get('score', 0.0)) 

    # Updated print statement to show original values
    print(f"Episode {episode_id} finished after {final_step_num} steps. Original Final Env Score: {final_score_from_env}, Original Total Reward: {total_reward_for_episode:.2f}, Original Total Perf Score: {total_perf_score_for_episode:.2f}")
    
    # Overwrite scores for Ace Attorney episodes
    effective_total_reward = total_reward_for_episode
    effective_total_perf_score = total_perf_score_for_episode
    effective_final_score_from_env = final_score_from_env

    # Record results with the adapter, using potentially overwritten values
    if hasattr(game_env, 'adapter') and game_env.adapter:
        game_env.adapter.record_episode_result(
            episode_id=episode_id,
            score=effective_final_score_from_env,       # Potentially overwritten
            steps=final_step_num,
            total_reward=effective_total_reward,        # Potentially overwritten
            total_perf_score=effective_total_perf_score # Potentially overwritten
        )
    else:
        print("Warning: game_env.adapter not found. Cannot record episode result for summary.")

    return

def main():
    prelim_parser = argparse.ArgumentParser(add_help=False)
    # No default for game_name here; it must be passed for prelim_parser to find the correct config.yaml
    prelim_parser.add_argument("--game_name", type=str, required=True, help="Game name needs to be passed to identify correct config.")
    prelim_parser.add_argument("--config_root_dir", type=str, default="Tetris-Benchmark/custom_04_tetris", help="Root path config files.")
    pre_args, remaining_argv = prelim_parser.parse_known_args()


    if not pre_args.game_name:
        print("Warning: --game_name not provided or not parsed by prelim_parser. Game-specific defaults from config.yaml might not be loaded.")
        config_dir_name = None # No specific game config can be loaded
    else:
        config_dir_name = game_config_mapping.get(pre_args.game_name.lower())
    
    if not config_dir_name and pre_args.game_name: # game_name was provided, but not in mapping
        print(f"Warning: Game name '{pre_args.game_name}' not found in game_config_mapping. Using game name directly for config path.")
        config_dir_name = pre_args.game_name
    elif not config_dir_name and not pre_args.game_name: # game_name wasn't provided to prelim_parser
        # Defaults_from_yaml will be empty, main parser will use its own defaults or fail on required args
        pass


    defaults_from_yaml = {}
    config_file_path = None # Initialize config_file_path to ensure it's always defined

    # Add game_name from prelim_parser to defaults_from_yaml so it's passed to set_defaults for the main parser
    if pre_args.game_name:
        defaults_from_yaml['game_name'] = pre_args.game_name

    if config_dir_name: # Only try to load if we have a config_dir_name
        config_file_path = os.path.join(pre_args.config_root_dir, config_dir_name, "config.yaml")
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as f:
                    loaded_yaml = yaml.safe_load(f)
                    if loaded_yaml:
                        if loaded_yaml.get('game_env'):
                            game_env_config_yaml = loaded_yaml['game_env']
                            defaults_from_yaml['num_runs'] = game_env_config_yaml.get('num_runs')
                            defaults_from_yaml['max_steps_per_episode'] = game_env_config_yaml.get('max_steps')
                            defaults_from_yaml['seed'] = game_env_config_yaml.get('seed')

                        if loaded_yaml.get('agent'):
                            agent_config_yaml = loaded_yaml['agent']

                            defaults_from_yaml['token_limit'] = agent_config_yaml.get('token_limit')
                            defaults_from_yaml['harness'] = agent_config_yaml.get('harness', False) # Default to False if not specified

                            defaults_from_yaml['model_name'] = agent_config_yaml.get('model_name')
                            defaults_from_yaml['observation_mode'] = agent_config_yaml.get('observation_mode')
                            defaults_from_yaml['use_custom_prompt'] = agent_config_yaml.get('use_custom_prompt')
                            defaults_from_yaml['use_reflection'] = agent_config_yaml.get('use_reflection')
                            defaults_from_yaml['use_perception'] = agent_config_yaml.get('use_perception')
                            defaults_from_yaml['use_summary'] = agent_config_yaml.get('use_summary')
                            defaults_from_yaml['scaffolding'] = agent_config_yaml.get('scaffolding')
                            
                            # Still load max_memory from its specific module config if present
                            if agent_config_yaml.get('modules'):
                                if agent_config_yaml['modules'].get('memory_module'):
                                    defaults_from_yaml['max_memory'] = agent_config_yaml['modules']['memory_module'].get('max_memory')
                        defaults_from_yaml = {k: v for k, v in defaults_from_yaml.items() if v is not None}
            except Exception as e:
                print(f"Warning: Could not load or process defaults from {config_file_path}: {e}")
        else:
            # This print is for when the specific game's config.yaml is not found
            print(f"Info: Game-specific config file {config_file_path} not found. Using command-line args and built-in defaults.")


    args = parse_arguments(defaults_map=defaults_from_yaml, argv_to_parse=remaining_argv)

    # Critical check: Ensure game_name has a value after all parsing.
    if not args.game_name:
        print("ERROR: game_name is missing after parsing. This should not happen if run.py provides it.")
        sys.exit(2) # Exit with a different code to distinguish from argparse error

    # Print information about which values are being used (command line has priority over config)
    for param_name, cli_value in args._cli_values.items():
        yaml_value = args._yaml_defaults.get(param_name)
        current_value = getattr(args, param_name)
        
        # Skip if no YAML value exists
        if yaml_value is None:
            continue

        # Special handling for model_name - always use CLI value
        if param_name == "model_name":
            if yaml_value != current_value:
                print(f"INFO: Using CLI value for model_name: {current_value} (YAML value ignored: {yaml_value})")
            continue

        # For other parameters, check if explicitly set on command line
        param_on_cli = f"--{param_name.replace('_', '-')}" in sys.argv
        if param_on_cli:
            if current_value != yaml_value:
                print(f"INFO: Using CLI value for '{param_name}': {current_value} (YAML value ignored: {yaml_value})")
        elif current_value != cli_value:
            print(f"INFO: Using YAML value for '{param_name}': {current_value} (CLI default was: {cli_value})")

    # params_where_config_wins = {
    #     'num_runs', 
    #     'max_steps_per_episode',
    #     'seed',
    #     'max_memory',
    #     'use_reflection',
    #     'use_perception',
    #     'use_summary',
    #     'scaffolding'
    # }

    # if config_file_path and os.path.exists(config_file_path):
    #     for param_name in params_where_config_wins:
    #         if param_name in defaults_from_yaml: # If the param was indeed in the loaded YAML config
    #             yaml_value = defaults_from_yaml[param_name]
    #             current_arg_value = getattr(args, param_name, None)
    #             if current_arg_value != yaml_value:
    #                 print(f"INFO: Overriding '{param_name}' with value from {config_file_path}. Was: {current_arg_value}, Now: {yaml_value}")
    #                 setattr(args, param_name, yaml_value)
    # --- End of override logic ---

    # Ensure agent_prompts_config_path uses the potentially overridden args.config_root_dir and correct config_dir_name
    # config_dir_name determined earlier is correct for the game specified by command line.
    final_config_dir_name = config_dir_name 
    if not final_config_dir_name and args.game_name: # If prelim parsing didn't get game_name but main args did
        final_config_dir_name = game_config_mapping.get(args.game_name.lower(), args.game_name)

    agent_prompts_config_path = None
    if final_config_dir_name: # It might still be None if game_name was never resolved
        agent_prompts_config_path = os.path.join(args.config_root_dir, final_config_dir_name, "module_prompts.json")
        if not os.path.isfile(agent_prompts_config_path):
            print(f"Warning: Agent prompts file {agent_prompts_config_path} not found. Agent will use default prompts.")
            agent_prompts_config_path = None
    else:
        print("Warning: Could not determine config directory for prompts due to missing game name resolution.")

    # DEBUG PRINT
    # print(f"DEBUG: Value of args.harness before check: {args.harness} (type: {type(args.harness)})")

    custom_modules_for_agent = None
    if args.harness:
        print("Initializing agent in HARNESS mode.")
        custom_modules_for_agent = {"perception_module": PerceptionModule, "reasoning_module": ReasoningModule}
    else:
        print("Initializing agent in NON-HARNESS (BaseModule) mode.")

    # --- Create Environment FIRST ---
    runner_log_dir_base = os.path.join("cache", args.game_name, args.model_name.replace("-", "_")[:15], datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(runner_log_dir_base, exist_ok=True)
    print(f"Agent and Environment cache directory: {runner_log_dir_base}")

    # Parse scaffolding parameter
    scaffolding_dict = None
    if args.scaffolding:
        try:
            if isinstance(args.scaffolding, dict):
                # New dictionary format from config
                funcname = args.scaffolding.get('funcname')
                funcArgs = args.scaffolding.get('funcArgs', {})
                
                # Map function names to actual function objects
                function_mapping = {
                    'draw_grid_on_image': draw_grid_on_image
                }
                
                if funcname in function_mapping:
                    scaffolding_dict = {
                        'func': function_mapping[funcname],
                        'funcArgs': funcArgs
                    }
                    print(f"Using scaffolding function: {funcname} with args: {funcArgs}")
                else:
                    print(f"Warning: Unknown scaffolding function '{funcname}'. Using None.")
            else:
                # Legacy tuple format for backward compatibility
                scaffolding_str = str(args.scaffolding).strip()
                if scaffolding_str.startswith('(') and scaffolding_str.endswith(')'):
                    scaffolding_str = scaffolding_str[1:-1]  # Remove parentheses
                parts = [int(x.strip()) for x in scaffolding_str.split(',')]
                if len(parts) == 2:
                    scaffolding_dict = {
                        'func': draw_grid_on_image,
                        'funcArgs': {'grid_dim': tuple(parts)}
                    }
                    print(f"Using legacy scaffolding grid: {tuple(parts)}")
                else:
                    print(f"Warning: Invalid scaffolding format '{args.scaffolding}'. Expected '(rows,cols)'. Using None.")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse scaffolding '{args.scaffolding}': {e}. Using None.")

    # Parse scaffolding parameter
    scaffolding_dict = None
    if args.scaffolding:
        try:
            if isinstance(args.scaffolding, dict):
                # New dictionary format from config
                funcname = args.scaffolding.get('funcname')
                funcArgs = args.scaffolding.get('funcArgs', {})
                
                # Map function names to actual function objects
                function_mapping = {
                    'draw_grid_on_image': draw_grid_on_image
                }
                
                if funcname in function_mapping:
                    scaffolding_dict = {
                        'func': function_mapping[funcname],
                        'funcArgs': funcArgs
                    }
                    print(f"Using scaffolding function: {funcname} with args: {funcArgs}")
                else:
                    print(f"Warning: Unknown scaffolding function '{funcname}'. Using None.")
            else:
                # Legacy tuple format for backward compatibility
                scaffolding_str = str(args.scaffolding).strip()
                if scaffolding_str.startswith('(') and scaffolding_str.endswith(')'):
                    scaffolding_str = scaffolding_str[1:-1]  # Remove parentheses
                parts = [int(x.strip()) for x in scaffolding_str.split(',')]
                if len(parts) == 2:
                    scaffolding_dict = {
                        'func': draw_grid_on_image,
                        'funcArgs': {'grid_dim': tuple(parts)}
                    }
                    print(f"Using legacy scaffolding grid: {tuple(parts)}")
                else:
                    print(f"Warning: Invalid scaffolding format '{args.scaffolding}'. Expected '(rows,cols)'. Using None.")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse scaffolding '{args.scaffolding}': {e}. Using None.")

    # --- Then Create Agent, passing the environment ---
    agent = BaseAgent(
        game_name=args.game_name,
        model_name=args.model_name,
        config_path=agent_prompts_config_path,
        harness=args.harness,
        use_custom_prompt=args.use_custom_prompt,
        max_memory=args.max_memory,
        use_reflection=args.use_reflection,
        use_perception=args.use_perception,
        use_summary=args.use_summary,
        custom_modules=custom_modules_for_agent,
        observation_mode=args.observation_mode,
        scaffolding=scaffolding_dict,
        cache_dir=runner_log_dir_base,
        vllm_url=args.vllm_url,
        modal_url=args.modal_url,
        token_limit=args.token_limit,
    )
    
    # runner_log_dir = agent.cache_dir # Agent already sets its cache_dir, this can be removed or used for verification
    # os.makedirs(runner_log_dir, exist_ok=True) # Already created by agent or above
    # print(f"Agent cache directory (contains episode logs and summary): {runner_log_dir}")

    # Env params are now loaded inside create_environment
    game_env = create_environment(
        game_name_arg=args.game_name,
        obs_mode_arg=args.observation_mode,
        config_dir_name_for_env_cfg=config_dir_name,
        cache_dir_for_adapter=runner_log_dir_base,
        harness=args.harness,
        multiagent_arg=args.multiagent_arg,
    )

    if game_env is None:
        print("Failed to create game environment. Exiting.")
        return

    for i in range(args.num_runs):
        run_id = i + 1
        # run_game_episode now doesn't return values, results are stored in adapter
        run_game_episode(agent, game_env, run_id, args)
        if i < args.num_runs - 1:
            print("Cooldown for 1 second before next run...")
            time.sleep(1)
    
    # Finalize and save summary using the adapter
    overall_stat_summary = {}
    if hasattr(game_env, 'adapter') and game_env.adapter:
        overall_stat_summary = game_env.adapter.finalize_and_save_summary(vars(args))
    else:
        print("Warning: game_env.adapter not found. Cannot finalize and save summary.")

    game_env.close() # Close environment after all runs

    print("\n" + "="*30 + " Overall Summary " + "="*30)
    print(f"Game: {args.game_name}, Model: {args.model_name}, Mode: {'Harness' if args.harness else 'BaseOnly'}, ObsMode: {args.observation_mode}")
    print(f"Number of runs: {args.num_runs}")
    
    if args.num_runs > 0 and overall_stat_summary:
        for key_snake, stats in overall_stat_summary.items():
            # Convert snake_case key back to Title Case for printing
            key_title = key_snake.replace("_", " ").title()
            if stats["mean"] is not None:
                print(f"Average {key_title}: {stats['mean']:.2f} (Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f})")
            else:
                print(f"Average {key_title}: N/A (no data)")
    else:
        print("No runs were completed or summary data is unavailable.")

if __name__ == "__main__":
    main() 