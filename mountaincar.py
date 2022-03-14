import gym
env = gym.make('MountainCar-v0')

observation = env.reset()
rewards = 0
for t in range(1000):
    env.render()
    #action = int(input())
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    rewards += reward
    print(observation, rewards)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        env.reset()
env.close()
