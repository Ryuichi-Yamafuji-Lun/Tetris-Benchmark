import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT)) 


from training_scripts.train_lin_grouped import QNetwork 

AGENT_DIR = Path(__file__).resolve().parent

MODEL_FILE_PATH = (
    AGENT_DIR.parent / "training_scripts" / "runs" / "train_lin_grouped" / "lambda_kappa__1__1760930295" / "train_lin_grouped.cleanrl_model"
)

MODEL_FILE_PATH_STR = str(MODEL_FILE_PATH)

class MockEnvs:
    """Mocks the essential parts of gym.vector.SyncVectorEnv needed by QNetwork constructor."""
    def __init__(self):
        self.single_observation_space = gym.spaces.Box(0.0, 200.0, (1, 13), np.float32)
        self.single_action_space = gym.spaces.Discrete(5)
        

DQN_MODEL = QNetwork(MockEnvs()).to('cpu')
DQN_MODEL.load_state_dict(torch.load(MODEL_FILE_PATH_STR, map_location=torch.device('cpu')))
DQN_MODEL.eval()

def agent_action(obs, info):
    """
    Retrieves the feature vector observation and returns the greedy action.
    """
    
    action_mask = info["action_mask"]
    num_actions = len(action_mask)

    q_values = torch.ones(num_actions, dtype=torch.float) * -np.inf

    valid_obs = torch.Tensor(obs[action_mask == 1]).to('cpu')

    with torch.no_grad():
        # The QNetwork outputs one value per valid observation
        calculated_q_values = DQN_MODEL(valid_obs).squeeze()

    # Place the calculated Q-values into the correct spots
    q_values[action_mask == 1] = calculated_q_values
    
    # 3. Choose the action with the highest Q-value
    action = torch.argmax(q_values).cpu().numpy()
    print(action)
    
    return action