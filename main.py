import gym
import Q_agent
import agent_train_test

env = gym.make('MountainCar-v0')

agent = Q_agent.Q_agent(env)

# 学習
learned_policy = agent_train_test.train(agent, env)

# テスト
for _ in range(10):
    print('reward: ', agent_train_test.test(agent, env, learned_policy))

env.close()