# random agent
import numpy as np

def agent_action(obs, info):
    action_mask = info["action_mask"]
    valid_actions = np.where(action_mask == 1)[0]
    return np.random.choice(valid_actions)