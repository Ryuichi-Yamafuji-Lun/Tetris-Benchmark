import argparse
import importlib.util

# Import the NormalizeRgbObservation wrapper from your training script
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# You'll need to add this class if it's not importable from train_ppo
class NormalizeRgbObservation(gym.ObservationWrapper):
    """Ensures RGB observations are in [0, 255] uint8 format."""

    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Box):
            shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )

    def observation(self, observation):
        if observation.dtype != np.uint8:
            if observation.max() <= 1.0 + 1e-5:
                observation = (observation * 255).astype(np.uint8)
            else:
                observation = np.clip(observation, 0, 255).astype(np.uint8)
        return observation


def load_agents(file_path):
    """Load agent function from a Python file."""
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        raise ImportError(f"Could not find module specification for file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.agent_action


def run_agent_test(agent, iterations=100, seed=1):
    """
    Run PPO agent test with RGB/image observations.
    Uses the same preprocessing pipeline as training.
    """
    # Create Tetris environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")

    # Apply same preprocessing as training
    env = RgbObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NormalizeRgbObservation(env)
    # env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)

    # Calculate scores
    reward_score = 0
    lines_cleared_score = 0

    # Run iterations
    for i in range(iterations):
        observation, info = env.reset(seed=seed + i)  # Different seed each iteration
        terminated = False
        truncated = False
        current_reward_score = 0
        current_lines_cleared_score = 0

        while not (terminated or truncated):
            action = agent(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)

            current_reward_score += reward
            # Handle clipped rewards - use episode stats if available
            if "episode" in info:
                current_lines_cleared_score = info["episode"].get("l", 0)
            else:
                current_lines_cleared_score += info.get("lines_cleared", 0)

        reward_score += current_reward_score
        lines_cleared_score += current_lines_cleared_score

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{iterations} iterations...")

    env.close()
    return [reward_score / iterations, lines_cleared_score / iterations]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tetris PPO agent benchmark.")

    parser.add_argument(
        "agents",
        type=str,
        nargs="+",
        help="List of PPO agent file paths (e.g., ppo_agent.py)",
    )

    parser.add_argument(
        "iterations",
        type=int,
        default=100,
        help="Number of games to run for each agent. Default = 100",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for testing. Default = 1"
    )

    args = parser.parse_args()

    # Run the benchmark
    print(f"\n{'='*60}")
    print(f"Running PPO Agent Benchmark")
    print(f"Iterations: {args.iterations} | Seed: {args.seed}")
    print(f"{'='*60}\n")

    for agent_path in args.agents:
        # Load the agent function
        agent_fn = load_agents(agent_path)

        # Get a clean name for logging
        agent_name = agent_path.split("/")[-1].replace(".py", "")

        print(f"--- Running Benchmark for: {agent_name} ---")

        # Run the agent
        avg_reward, avg_lines = run_agent_test(agent_fn, args.iterations, args.seed)

        print(f"\n{agent_name} Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Lines Cleared: {avg_lines:.2f}")
        print(f"{'='*60}\n")
