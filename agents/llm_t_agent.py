import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_agent.llm_tetris_handler import agent_action as llm_handler_action
from llm_agent.llm_tetris_handler import AGENT_INSTANCE 

def agent_action(obs, info):
    
    return llm_handler_action(obs, info)