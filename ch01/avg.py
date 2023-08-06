import numpy as np

np.random.seed(0)


def simple():
    rewards = []

    for n in range(1, 11):
        reward = np.random.rand()
        rewards.append(reward)
        Q = sum(rewards) / n
        print(Q)


def better():
    Q = 0

    for n in range(1, 11):
        reward = np.random.rand()
        # Q = Q + (reward - Q) / n
        Q += (reward - Q) / n
        print(Q)


# simple()
better()
