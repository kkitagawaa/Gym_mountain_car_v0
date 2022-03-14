import numpy as np

from setting import MAX_STEPS, NUM_EPISODES

def train(agent, env):
    # ベスト報酬
    best_reward = -float('inf')

    # 学習ループ
    for episode in range(NUM_EPISODES):
        # 環境のリセット
        state = env.reset()
        done = False
        total_reward = 0.0

        # 1エピソードのループ
        for step in range(MAX_STEPS):
            # 行動の取得
            action = agent.get_action(state)

            # １ステップの実行
            next_state, reward, done, info = env.step(action)

            # 学習
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # エピソード完了
            if done:
                break
        
        # ベスト報酬の更新
        if total_reward > best_reward:
            best_reward = total_reward
        print("episode:{} reward:{} best_reward:{} eps:{}".format(episode, total_reward, best_reward, agent.epsilon))
    
    # ポリシーを返す
    #print(np.argmax(agent.Q, axis=2))
    return np.argmax(agent.Q, axis=2)

def test(agent, env, policy):
    # 環境のリセット
    obs = env.reset()
    done = False
    total_reward = 0.0

    # 1エピソードのループ
    for step in range(MAX_STEPS):
        # 環境の描画
        env.render()

        # 行動の取得
        action = policy[agent.discretize(obs)]

        # 1ステップの実行
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward

        # エピソード完了
        if done:
            break
    
    # 報酬和を返す
    return total_reward