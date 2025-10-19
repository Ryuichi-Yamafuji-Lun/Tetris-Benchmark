import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    # creates tetris env
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    # ensures that blocks are the same for testing
    env.reset(seed=42)

    terminated = False
    while not terminated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(info["lines_cleared"])
    print("Game Over!")