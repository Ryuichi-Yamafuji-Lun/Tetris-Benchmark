# 🕹️ Tetris AI Benchmarking Harness

This repository hosts a **standardized benchmarking harness** designed to empirically compare the strategic performance of AI agents in the game of **Tetris**.

---

## 🔍 Key Features

- **Dynamic Agent Loading**  
  Agents are dynamically loaded from file paths specified at runtime using `argparse` and `importlib.util`, allowing flexible testing of multiple models in a single command.

- **Reproducible Environment**  
  A fixed random seed (`seed=1`) ensures all agents play the **identical sequence of Tetris pieces** for fair, objective comparison.

- **Dual Metric Output**  
  Tracks both:
  - **Total Accumulated Reward** — the RL training metric  
  - **Total Lines Cleared** — the objective game proficiency metric

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
# .venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
This project requires `gymnasium`, the `tetris_gymnasium` environment, and `pandas` for data handling.
```bash
pip install gymnasium pandas tetris-gymnasium
```

> **Note:** Additional dependencies (e.g., `stable-baselines3`, OpenAI API libraries) are required to run the RL and LLM agents.

---

## 🚀 Usage: Running the Benchmark

The main benchmark runner is **`test.py`**.  
All agent files must be located in the `./agents/` directory and must expose their logic in a function named:

```python
agent_action(env)
```

### 🧭 Command Structure
```bash
python3 test.py [AGENT_FILE_1] [AGENT_FILE_2] ... [ITERATIONS]
```

### 🧪 Examples

| Goal | Command | Description |
|------|----------|-------------|
| **Baseline Run (Single Agent)** | `python3 test.py ./agents/random_agent.py 100` | Runs the Random Agent for 100 games. |
| **Head-to-Head Comparison** | `python3 test.py ./agents/random_agent.py ./agents/rl_agent.py 50` | Runs the Random Agent and RL Agent sequentially (50 games each) under identical piece sequences and reports averages. |

### 🖥️ Example Console Output
```
--- Running Benchmark for: random_agent ---
Random Agent: Avg Reward Score: 180 | Avg Lines Cleared: 15
```

---

## 📊 Benchmark Metrics

The benchmark reports the **average** of two distinct performance scores over all iterations:

| Metric | Calculation | Purpose in Research |
|---------|--------------|---------------------|
| **Avg Reward Score** | Total accumulated reward (Σ rₜ) per episode | The RL policy metric — measures success in maximizing its core optimization signal. |
| **Avg Lines Cleared** | Total lines cleared per episode (Σ info['lines_cleared']) | The objective game metric — measures universal strategic proficiency for fair comparison between RL and LLM agents. |

---

## 🧩 Environment Credits

This project builds upon the **Tetris Gymnasium** environment by Max-We and Philipp Hartl — a modular and fully configurable Reinforcement Learning environment built on the Gymnasium API.

**Citation:**
```bibtex
@booklet{EasyChair:13437,
  author = {Maximilian Weichart and Philipp Hartl},
  title = {Piece by Piece: Assembling a Modular Reinforcement Learning Environment for Tetris},
  howpublished = {EasyChair Preprint 13437},
  year = {EasyChair, 2024}
}
```

-
