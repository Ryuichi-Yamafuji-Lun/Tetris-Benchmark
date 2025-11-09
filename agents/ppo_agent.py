import torch
import numpy as np
import gymnasium as gym
from training_scripts.train_ppo import Agent
from pathlib import Path
import sys
import glob
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

AGENT_DIR = Path(__file__).resolve().parent

# Automatically find the most recent model
model_files = glob.glob(
    str(AGENT_DIR.parent / "training_scripts" / "runs" / "**" / "train_ppo.cleanrl_model"), 
    recursive=True
)
if not model_files:
    raise FileNotFoundError("No trained PPO model found! Please train the model first.")

MODEL_FILE_PATH_STR = max(model_files, key=os.path.getctime)
print(f"Loading model from: {MODEL_FILE_PATH_STR}")

class MockEnvs:
    """Mocks the essential parts of gym.vector.SyncVectorEnv needed by Agent constructor."""
    def __init__(self):
        # Match the actual training: (40, 13)
        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=200.0, shape=(40, 13), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Discrete(40)

PPO_MODEL = Agent(MockEnvs()).to('cpu')
PPO_MODEL.load_state_dict(torch.load(MODEL_FILE_PATH_STR, map_location=torch.device('cpu')))
PPO_MODEL.eval()

def agent_action(obs, info):
    """
    Retrieves the observation and returns the action from PPO policy.
    """
    obs_tensor = torch.Tensor(obs).unsqueeze(0).to('cpu')
    
    with torch.no_grad():
        action, _, _, _ = PPO_MODEL.get_action_and_value(obs_tensor)
    
    return action.cpu().numpy()[0]