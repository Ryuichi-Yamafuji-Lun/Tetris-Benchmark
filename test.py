import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

from agents import random_agent
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
    # Run the benchmark
    TEST_SEED = 1
    TEST_ITERATIONS = 100

    # run random agent
    random_agent_reward_score, random_agent_lines_cleared_score = run_agent_test(random_agent.agent_action_random, TEST_ITERATIONS, TEST_SEED)
    print(f"Random Agent: Avg Reward Score: {random_agent_reward_score} | Avg Lines Cleared: {random_agent_lines_cleared_score}")