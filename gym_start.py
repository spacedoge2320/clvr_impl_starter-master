import gym
import sprites_env.__init__
import cv2
import time

# 환경 생성 및 초기화

env = gym.make('Sprites-v1')
observation = env.reset()
print(1)

# 환경을 몇 번의 스텝 동안 실행
for _ in range(1000):
    action = env.action_space.sample()  # 무작위로 액션 선택
    observation, reward, done, info = env.step(action)  # 액션 수행 및 결과 받기
    print(f"reward: {reward}, done: {done}, info: {info}")
    observation_resized = cv2.resize(observation, (500, 500))
    cv2.imshow('Environment', observation_resized)
    cv2.waitKey(1)
    time.sleep(0.5)
    if done:
        observation = env.reset()  # 게임 종료 시 재시작
        print('done')

env.close()  # 환경 종료


