import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from training_scripts.train_ppo import Agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

AGENT_DIR = Path(__file__).resolve().parent

# Update this path to your trained PPO model
MODEL_FILE_PATH = (
    AGENT_DIR.parent
    / "runs"
    / "tetris_gymnasium"
    / "Tetris__train_ppo__1__1763520300"
    / "train_ppo.cleanrl_model"
)

MODEL_FILE_PATH_STR = str(MODEL_FILE_PATH)


class MockEnvs:
    """Mocks the essential parts of gym.vector.SyncVectorEnv needed by Agent constructor."""

    def __init__(self):
        # PPO uses 4 stacked grayscale frames of 84x84 pixels
        self.single_observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        self.single_action_space = gym.spaces.Discrete(
            8
        )  # Adjust based on your Tetris action space


PPO_MODEL = Agent(MockEnvs()).to("cpu")
PPO_MODEL.load_state_dict(
    torch.load(MODEL_FILE_PATH_STR, map_location=torch.device("cpu"))
)
PPO_MODEL.eval()


def agent_action(obs, info):
    """
    Takes image observation (4 stacked grayscale frames) and returns action.

    Args:
        obs: numpy array of shape (4, 84, 84) - 4 stacked grayscale frames
        info: dict with environment info

    Returns:
        action: integer action
    """
    # Convert observation to tensor and add batch dimension
    obs_tensor = torch.Tensor(obs).unsqueeze(0).to("cpu")

    with torch.no_grad():
        # Get action logits from policy network
        hidden = PPO_MODEL.network(obs_tensor / 255.0)
        logits = PPO_MODEL.actor(hidden).squeeze()

        # Choose action with highest probability (deterministic evaluation)
        action = torch.argmax(logits).cpu().numpy()

    return int(action)
