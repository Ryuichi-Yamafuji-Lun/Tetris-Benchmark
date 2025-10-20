import pandas as pd
import argparse
import importlib.util
from pathlib import Path
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

# agent imports 
from agents import random_agent

# loading agents
def load_agents(file_path):
    module_name = Path(file_path).stem

    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        raise ImportError(f"Could not find module specification for file: {file_path}")
    
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module.agent_action

# benchmarking function
def run_agent_test(agent, iterations = 100, seed = 1):
    
    # creates tetris env
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")

    # calculate scores
    reward_score = 0
    lines_cleared_score = 0

    # iterate through x times
    for _ in range(iterations):
        # reset the tetris environment
        env.reset(seed=seed)

        terminated = False
        current_reward_score = 0
        current_lines_cleared_score = 0
        while not terminated:

            # agent chooses the action
            action = agent(env)

            # environment steps
            observation, reward, terminated, truncated, info = env.step(action)

            # update current_score
            current_reward_score += reward
            current_lines_cleared_score += info["lines_cleared"]

        reward_score += current_reward_score
        lines_cleared_score += current_lines_cleared_score
    
    env.close()
    return [reward_score / iterations, lines_cleared_score / iterations]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tetris Gymnasium benchmark.")

    parser.add_argument(
        'agents',
        type=str,
        nargs='+',
        help='List of agent file paths (e.g., random_agent.py rl_agent.py)'
    )

    parser.add_argument(
        'iterations',
        type=int,
        default=100,
        help='Number of games to run for each agent. Default = 100'
    )

    args = parser.parse_args()

    # Run the benchmark
    TEST_SEED = 1

    for agent_path in args.agents:
        # Load the agent function
        agent_fn = load_agents(agent_path)
        
        # Get a clean name for logging
        agent_name = agent_path.split('/')[-1].replace('.py', '')
        
        print(f"--- Running Benchmark for: {agent_name} ---")
        
        # Run the agent using the parsed arguments
        avg_reward, avg_lines = run_agent_test(
            agent_fn, 
            args.iterations, 
            TEST_SEED
        )
        
        print(f"{agent_name}: Avg Reward: {avg_reward} | Avg Lines: {avg_lines}")