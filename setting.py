# 定数(訓練)
NUM_EPISODES = 30000 # 学習するエピソード数
MAX_STEPS = 200 # 1エピソードの最大ステップ数

# 定数(学習アルゴリズム)
EPSILON_MIN = 0.005 # εの最小値
EPSILON_INIT = 1.0 # εの初期値
EPSILON_DECAY = 500 * EPSILON_MIN/(NUM_EPISODES*MAX_STEPS) # ε値の減衰量
ALPHA = 0.05 # 学習係数（1回の学習の更新の大きさ）
GAMMA = 0.98 # 時間割引率（0〜1）
NUM_BINS = 30 # 離散化時の分割数