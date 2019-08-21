# import gym
#
# env = gym.make("CartPole-v1")
# observation = env.reset()
# for _ in range(100000):
#     env.render()
#     action = env.action_space.sample()  # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#
#     if done:
#         observation = env.reset()
# env.close()

import gym
env = gym.make('HalfCheetah-v2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("reward:{}".format(reward))
        print("info:{}".format(reward))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()