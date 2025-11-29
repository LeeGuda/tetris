import tensorflow as tf
from collections import deque
import numpy as np
import random
import pygame
import os
import time
import subprocess
import threading
from multiprocessing import Process, Manager, Lock, cpu_count
from TetrisEnv import TetrisEnv
from PrioritizedReplayMemory import PrioritizedReplayMemory

AdamOptimizer = tf.keras.optimizers.Adam

# --- 전역 상수 설정 ---
STATE_SIZE = 14 
ACTION_MAP = [(r, x) for r in range(4) for x in range(10)]
ACTION_SIZE = len(ACTION_MAP)

REPLAY_MEMORY_SIZE = 400000 
N_WORKERS = cpu_count() - 1 

# 모델 및 메모리 저장 경로
MODEL_SAVE_PATH = 'dqn_tetris_weights.weights.h5' 
MEMORY_SAVE_PATH = 'prioritized_replay_memory.pkl' 

# 깃허브 푸시 함수 (두 파일 모두 처리)
def git_push_thread(step):
    try:
        print(f"\n[Git] Uploading files to GitHub...")
        
        # 1. git add: 모델 파일과 메모리 파일 모두 추가
        subprocess.run(["git", "add", MODEL_SAVE_PATH], check=True, capture_output=True)
        subprocess.run(["git", "add", MEMORY_SAVE_PATH], check=True, capture_output=True) 
        
        # 2. git commit
        commit_message = f"Auto-save: Weights & PER Memory at step {step}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True)
        
        # 3. git push
        subprocess.run(["git", "push"], check=True, capture_output=True)
        
        print(f"[Git] Successfully pushed to GitHub at step {step}!")
    except subprocess.CalledProcessError as e:
        print(f"[Git] Push skipped or failed: {e.output.decode()}")

class DQNAgent:
    """중앙 및 모니터링 에이전트"""
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """CPU에서 동작하는 신경망 구축"""
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=AdamOptimizer(learning_rate=self.learning_rate))
            return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def save_weights(self, filename):
        with tf.device('/cpu:0'):
            self.model.save_weights(filename)

    def load_weights(self, filename):
        with tf.device('/cpu:0'):
            self.model.load_weights(filename)
            self.update_target_model()

    def act(self, state, possible_actions, ACTION_MAP, epsilon=0.0):
        # 1. 탐험
        if np.random.rand() <= epsilon:
            random_action_tuple = random.choice(possible_actions)
            try:
                action_index = ACTION_MAP.index(random_action_tuple)
            except ValueError:
                action_index = random.randrange(self.action_size)
            return action_index
        
        # 2. 활용
        with tf.device('/cpu:0'):
            state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
            q_values_tensor = self.model(state_tensor, training=False)
            q_values = q_values_tensor.numpy()[0]
        
        possible_indices = {ACTION_MAP.index(act) for act in possible_actions if act in ACTION_MAP}
        
        for i in range(self.action_size):
            if i not in possible_indices:
                q_values[i] = -1e9
                
        action_index = np.argmax(q_values)
        return action_index

    def replay(self, memory, batch_size): 
        if memory.size() < batch_size * 4:
            return

        states, action_indices, rewards, next_states, dones, indices, is_weights = memory.sample(batch_size)
        
        with tf.device('/cpu:0'):
            # Double DQN Logic
            next_actions_indices = np.argmax(self.model(next_states, training=False).numpy(), axis=1)
            next_q_values_target = self.target_model(next_states, training=False).numpy()
            max_next_q_values = next_q_values_target[np.arange(batch_size), next_actions_indices]
        
            targets = rewards + self.gamma * max_next_q_values * (1 - dones.astype(int))
            target_f = self.model(states, training=False).numpy() 
            
            # 예측 Q 값과 목표 Q 값의 차이 (TD 오차) 계산
            td_errors = targets - target_f[np.arange(batch_size), action_indices]
            
            # PER: TD 오차를 사용하여 우선순위 업데이트
            memory.update_priorities(indices, td_errors)
        
            # 모델 업데이트
            for i in range(batch_size):
                target_f[i, action_indices[i]] = targets[i]
            
            # PER: Importance Sampling Weight를 반영하여 손실을 계산
            is_weights = is_weights.reshape(batch_size,) 
            
            # train_on_batch 사용
            self.model.train_on_batch(states, target_f, sample_weight=is_weights)


def worker_process(worker_id, memory_queue, shared_weights, epsilon_map, global_steps, lock):
    """학습 워커 프로세스 (경험 수집 역할)"""
    env = TetrisEnv(render_mode='none') 
    local_agent = DQNAgent()
    local_agent.set_weights(shared_weights) 
    
    print(f"Worker {worker_id} started. Initial Epsilon: {epsilon_map['epsilon']:.4f}")
    
    episode_count = 0 
    SYNC_FREQ = 5 
    
    while True:
        if episode_count % SYNC_FREQ == 0:
            local_agent.set_weights(shared_weights)
        
        state = env.reset()
        done = False
        
        while not done:
            epsilon = epsilon_map['epsilon']
            possible_actions = env.get_possible_actions()
            action_index = local_agent.act(state, possible_actions, ACTION_MAP, epsilon=epsilon)
            action = ACTION_MAP[action_index]
            
            next_state, reward, done, _ = env.step(action)
            
            # 워커는 경험과 초기 높은 우선순위를 메인 프로세스로 전달
            memory_queue.put((state, action_index, reward, next_state, done))
            
            with lock:
                global_steps['value'] += 1
            
            state = next_state
        
        with lock:
            if epsilon_map['epsilon'] > 0.05: 
                epsilon_map['epsilon'] *= 0.99995 
        
        episode_count += 1 


def distributed_train_dqn(episodes=50000, batch_size=128, target_update_freq=10, render_freq=5, worker_count=N_WORKERS):
    
    global_agent = DQNAgent() 
    
    # 1. 가중치 로드
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading previous weights from {MODEL_SAVE_PATH}...")
        try:
            global_agent.load_weights(MODEL_SAVE_PATH)
            print("Weights successfully loaded. Resuming training.")
        except Exception as e:
            print(f"Error loading weights ({e}). Starting training from scratch.")

    # 2. PER 메모리 및 매니저 초기화
    memory = PrioritizedReplayMemory(REPLAY_MEMORY_SIZE, STATE_SIZE)
    memory.load_memory(MEMORY_SAVE_PATH)
    
    manager = Manager()
    memory_queue = manager.Queue()
    
    shared_weights = manager.list(global_agent.get_weights())
    epsilon_map = manager.dict({'epsilon': 1.0})
    global_steps = manager.dict({'value': 0})
    lock = manager.Lock()

    monitor_agent = DQNAgent() 
    monitor_env = TetrisEnv(render_mode='human')
    monitor_state = monitor_env.reset()
    monitor_done = False
    monitor_total_reward = 0.0
    monitor_step_count = 0
    
    # 워커 시작
    print(f"\n--- Starting Distributed Training with {worker_count} Workers (CPU Mode) ---")
    workers = []
    actual_worker_count = max(1, worker_count)
    for i in range(actual_worker_count):
        p = Process(target=worker_process, args=(i, memory_queue, shared_weights, epsilon_map, global_steps, lock))
        workers.append(p)
        p.start()

    # 메인 루프
    global_train_count = 0
    
    while global_train_count < episodes:
        
        # 큐에 쌓인 데이터를 PER 메모리에 저장
        while not memory_queue.empty():
            experience = memory_queue.get()
            memory.store(experience, memory.max_priority if memory.size() > 0 else 1.0) 
            
        if memory.size() < batch_size * 4: 
            print(f"Waiting for experience... Current size: {memory.size()}", end='\r')
            time.sleep(1)
            continue
            
        global_agent.replay(memory, batch_size)
        global_train_count += 1
        
        if global_train_count % target_update_freq == 0:
            global_agent.update_target_model()
        
        if global_train_count % 1 == 0: 
             new_weights = global_agent.get_weights()
             for i, w in enumerate(new_weights):
                 shared_weights[i] = w
        
        # [수정] 주기적 저장 (1000 스텝마다) - 푸시 기능은 삭제됨
        if global_train_count % 1000 == 0 and global_train_count > 0:
            print(f"\n--- Saving model/memory weights at Train Step {global_train_count} ---")
            global_agent.save_weights(MODEL_SAVE_PATH)
            memory.save_memory(MEMORY_SAVE_PATH) 
            
            # Git 푸시 스레드 시작 코드가 삭제되었습니다.
            
        # 렌더링 및 모니터링
        if global_train_count % render_freq == 0: 
            monitor_agent.set_weights(global_agent.get_weights())
            
            if monitor_done:
                monitor_state = monitor_env.reset()
                monitor_total_reward = 0.0 
                monitor_step_count = 0
                monitor_done = False
            
            possible_actions = monitor_env.get_possible_actions()
            action_index = monitor_agent.act(monitor_state, possible_actions, ACTION_MAP, epsilon=0.0)
            action = ACTION_MAP[action_index]
            
            monitor_state, reward, monitor_done, _ = monitor_env.step(action)
            monitor_total_reward += reward
            monitor_step_count += 1
            
            monitor_env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nUser quit signal received. Saving and Exiting...")
                    global_agent.save_weights(MODEL_SAVE_PATH) 
                    memory.save_memory(MEMORY_SAVE_PATH) 
                    
                    # [추가] 종료 시 Git 푸시
                    git_push_thread(global_train_count)
                    
                    monitor_env.close()
                    for p in workers: p.terminate(); p.join()
                    return

        if global_train_count % 10 == 0:
            print(f"Train Step: {global_train_count}/{episodes}, Global Steps: {global_steps['value']}, Epsilon: {epsilon_map['epsilon']:.4f} | Monitor Reward: {monitor_total_reward:.2f}")

    print("\n--- Distributed Training Finished. Saving Model Weights ---")
    global_agent.save_weights(MODEL_SAVE_PATH)
    memory.save_memory(MEMORY_SAVE_PATH)
    
    # [추가] 최종 푸시
    git_push_thread(global_train_count)
    
    monitor_env.close()
    for p in workers:
        p.terminate()
        p.join()
        
    print("Training Complete.")

if __name__ == '__main__':
    pygame.init() 
    distributed_train_dqn(episodes=50000, batch_size=128, target_update_freq=10, render_freq=5, worker_count=N_WORKERS)