import numpy as np

from setting import ALPHA, EPSILON_DECAY, EPSILON_INIT, EPSILON_MIN, GAMMA, NUM_BINS

class Q_agent(object):
    
    def __init__(self, env) -> None:
        # 状態空間と行動空間の値
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.action_shape = env.action_space.n

        # 離散化の値の準備
        self.bin_width = (self.obs_high - self.obs_low)/NUM_BINS

        # Q関数の生成(31*31*3)
        self.Q = np.zeros((NUM_BINS+1, NUM_BINS+1, self.action_shape))

        # εの初期値
        self.epsilon = EPSILON_INIT

    # 状態を連続値から離散値に変換
    def discretize(self, state):
        return tuple(((state-self.obs_low)/self.bin_width).astype(int))

    # 行動選択
    def get_action(self, state):
        # 状態を連続値から離散値に変換
        state = self.discretize(state)

        # εの減衰
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY

        # ε-greedy方策
        # 最良な行動の選択(確率1-ε)
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        # ランダム行動の選択
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    # Q関数を更新
    def learn(self, state, action, reward, next_state):
        # 状態を連続値から離散値に変換
        state = self.discretize(state)
        next_state = self.discretize(next_state)

        # Q関数の更新
        td_target = reward + GAMMA * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += ALPHA * td_error
