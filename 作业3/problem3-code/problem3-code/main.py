import random
from collections import defaultdict
import numpy as np

from cliff_walking import Env

# 参数
epsilon = 0.1
learningRate = 0.5
discountFactor = 1


def getBestAction(qTable, ob, envs):
    stateQ = qTable[str(ob)]
    actions = envs.getStateAction(ob)
    maxList = []
    maxValue = stateQ[actions[0]]
    maxList.append(actions[0])
    for i in actions[1:]:
        if stateQ[i] > maxValue:
            maxList.clear()
            maxValue = stateQ[i]
            maxList.append(i)
        elif stateQ[i] == maxValue:
            maxList.append(i)
    return random.choice(maxList), maxValue


def qLearning():
    # Q值矩阵
    qTable = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    envs = Env(3, 0)

    for i in range(0, 10000):
        ob = envs.getPosition()
        if np.random.rand() < epsilon:
            action = random.choice(envs.getStateAction(ob))
        else:
            action, _ = getBestAction(qTable, ob, envs)

        next_ob, transition = envs.transition(action)
        flag = (transition == -100 or next_ob == 47)
        oldQ = qTable[str(ob)][action]
        if flag:
            newQ = transition
        else:
            _, maxQ = getBestAction(qTable, next_ob, envs)
            newQ = transition + discountFactor * maxQ
        qTable[str(ob)][action] += learningRate * (newQ - oldQ)
        if flag:
            envs.reset()

    envs.reset()
    ob = envs.getPosition()
    path = [ob]
    while True:
        action, _ = getBestAction(qTable, ob, envs)
        ob, _ = envs.transition(action)
        path.append(ob)
        if ob == 47:
            break

    print("Qlearning:")
    print(path)


def sarsa():
    # Q值矩阵
    qTable = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    envs = Env(3, 0)

    ob = envs.getPosition()
    action, _ = getBestAction(qTable, ob, envs)
    for i in range(0, 100000):
        next_ob, transition = envs.transition(action)
        flag = (transition == -100 or next_ob == 47)
        if np.random.rand() < epsilon:
            action_ = random.choice(envs.getStateAction(next_ob))
        else:
            action_, _ = getBestAction(qTable, next_ob, envs)
        oldQ = qTable[str(ob)][action]
        if flag:
            newQ = transition
        else:
            newQ = transition + discountFactor * qTable[str(next_ob)][action_]
        qTable[str(ob)][action] += learningRate * (newQ - oldQ)
        ob = next_ob
        action = action_
        if flag:
            envs.reset()
            ob = envs.getPosition()
            action, _ = getBestAction(qTable, ob, envs)

    envs.reset()
    ob = envs.getPosition()
    path = [ob]
    while True:
        action, _ = getBestAction(qTable, ob, envs)
        ob, _ = envs.transition(action)
        path.append(ob)
        if ob == 47:
            break

    print("Sarsa:")
    print(path)


if __name__ == "__main__":
    sarsa()
    qLearning()
