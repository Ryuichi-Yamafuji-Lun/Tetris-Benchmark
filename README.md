# 🕹️ Tetris AI Benchmarking Harness

This repository hosts a **standardized benchmarking harness** designed to empirically compare the performance of different AI agents in the game of **Tetris**.  

---

## 🔍 Key Features

- **Dynamic Agent Loading**  
  Test multiple agents in a single command; the harness dynamically loads them from file paths.

- **Reproducible Environment**  
  A fixed random seed ensures all agents face the identical sequence of Tetris pieces for a fair, objective comparison.

- **Sophisticated State Representation**  
  The environment is wrapped with `GroupedActionsObservations` and `FeatureVectorObservation` to provide agents with a **rich, structured observation space** and an `action_mask` for identifying valid moves.

- **Dual Metric Output**  
  Tracks both:
  - **Accumulated Reward** (for RL policy evaluation)  
  - **Total Lines Cleared** (for objective proficiency)

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/RyuichiLun/Tetris-Benchmark.git
cd Tetris-Benchmark
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate     # On macOS/Linux
# .venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies
This project requires `gymnasium` and the `tetris_gymnasium` environment.
```bash
pip install gymnasium pandas tetris-gymnasium
```

> **Note:** Agents may have additional dependencies (e.g., `torch` for the RL agent or `openai` for an LLM agent). Please install them as needed.

---

## 🚀 Usage: Running the Benchmark & Adding Agents

The main benchmark runner is **`test.py`**.

### 🧠 Creating a Custom Agent

All agent files must be located in the `./agents/` directory.  
Each agent must expose its logic in a function with the following signature:

```python
def agent_action(obs, info):
    ...
```

#### Parameters:
- **obs (`np.array`)** — A structured NumPy array containing the feature vectors for the currently valid moves.  
  The shape of this array corresponds to the number of 1s in the `action_mask`.
- **info (`dict`)** — Contains critical metadata about the current state. Your agent will primarily use the `action_mask` from this dictionary to make valid decisions.

Example `info` dictionary:
```python
{
    'lines_cleared': 0,                            # Lines cleared in the last step
    'action_mask': np.array([1., 1., ..., 0.]),   # Binary mask of all possible actions (1=valid, 0=invalid)
    'action_space_size': 40,                       # Total size of the action space
}
```

---

## 🧭 Command Structure
```bash
python3 test.py [AGENT_FILE_1] [AGENT_FILE_2] ... [ITERATIONS]
```

### 🧪 Examples

| Goal | Command | Description |
|------|----------|-------------|
| **Baseline Run (Single Agent)** | `python3 test.py ./agents/random_agent.py 100` | Runs the Random Agent for 100 games. |
| **Head-to-Head Comparison** | `python3 test.py ./agents/random_agent.py ./agents/dqn_agent.py 100` | Runs the Random and DQN agents sequentially for 100 games each, using the same piece sequences for fair comparison. |

### 🖥️ Example Console Output
```
--- Running Benchmark for: dqn_agent ---
dqn_agent: Avg Reward: 605.04 | Avg Lines: 29.01
```

---

## 📊 Benchmark Metrics

The benchmark reports the **average of two distinct performance scores** over all iterations.  
`lines_cleared` is extracted directly from the environment's `info` dictionary at each step.

| Metric | Calculation | Purpose in Research |
|---------|--------------|---------------------|
| **Avg Reward** | Total accumulated reward (Σ rₜ) per episode | The RL policy metric — measures success in maximizing the agent's core optimization signal. |
| **Avg Lines Cleared** | Total lines cleared per episode (Σ info['lines_cleared']) | The objective game metric — measures universal strategic proficiency for fair comparison between all agents. |

---

## 🧩 Environment Credits

This project builds upon the **Tetris Gymnasium** environment by *Maximilian Weichart* and *Philipp Hartl* — a modular and fully configurable Reinforcement Learning environment built on the Gymnasium API.

**Citation:**
```bibtex
@booklet{EasyChair:13437,
  author = {Maximilian Weichart and Philipp Hartl},
  title = {Piece by Piece: Assembling a Modular Reinforcement Learning Environment for Tetris},
  howpublished = {EasyChair Preprint 13437},
  year = {EasyChair, 2024}
}
```
 
**Course:** CSCI 566 – Deep Learning and its Applications  
**Institution:** University of Southern California
