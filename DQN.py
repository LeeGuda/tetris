import tensorflow as tf
from collections import deque
import numpy as np
import random
import pygame
import os
import time
from multiprocessing import Process, Queue, Manager, cpu_count
from TetrisEnv import TetrisEnv

AdamOptimizer = tf.keras.optimizers.Adam

# --- 전역 상수 설정 ---
# 💡 TetrisEnv에서 상태 벡터가 1개 추가되었으므로 14로 설정
STATE_SIZE = 14 
ACTION_MAP = [(r, x) for r in range(4) for x in range(10)]
ACTION_SIZE = len(ACTION_MAP)
REPLAY_MEMORY_SIZE = 5000000
# 💡 7800X3D 활용 (코어 수 - 1)
N_WORKERS = cpu_count() - 1 

# 모델 저장 경로
MODEL_SAVE_PATH = 'dqn_tetris_weights.weights.h5' 

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

    def replay(self, memory_queue, batch_size):
        if memory_queue.qsize() < batch_size:
            return

        batch = []
        while not memory_queue.empty() and len(batch) < batch_size:
            batch.append(memory_queue.get())
            
        if not batch: return
        actual_batch_size = len(batch)

        states = np.array([e[0] for e in batch])
        action_indices = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        with tf.device('/cpu:0'):
            # Double DQN Logic
            next_actions_indices = np.argmax(self.model(next_states, training=False).numpy(), axis=1)
            next_q_values_target = self.target_model(next_states, training=False).numpy()
            max_next_q_values = next_q_values_target[np.arange(actual_batch_size), next_actions_indices]
        
            targets = rewards + self.gamma * max_next_q_values * (1 - dones.astype(int))
            target_f = self.model(states, training=False).numpy() 
        
            for i in range(len(batch)):
                target_f[i, action_indices[i]] = targets[i]
        
            self.model.train_on_batch(states, target_f)


def worker_process(worker_id, memory_queue, shared_weights, epsilon_map, global_steps, lock):
    """학습 워커 프로세스"""
    # 렌더링 없음 (None)
    env = TetrisEnv(render_mode='none') 
    local_agent = DQNAgent()
    local_agent.set_weights(shared_weights) 
    
    print(f"Worker {worker_id} started. Initial Epsilon: {epsilon_map['epsilon']:.4f}")
    
    episode_count = 0 
    SYNC_FREQ = 5 # 가중치 동기화 주기
    
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
            
            memory_queue.put((state, action_index, reward, next_state, done))
            
            with lock:
                global_steps['value'] += 1
            
            state = next_state
        
        # [핵심 수정] Epsilon Decay 정책 변경 (매우 천천히 감소)
        with lock:
            if epsilon_map['epsilon'] > 0.05: # 최소 탐험 확률 5% 유지
                epsilon_map['epsilon'] *= 0.99995 
        
        episode_count += 1 


def distributed_train_dqn(episodes=50000, batch_size=128, target_update_freq=10, render_freq=5, worker_count=N_WORKERS):
    
    global_agent = DQNAgent() 
    
    # 저장된 가중치 로드
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading previous weights from {MODEL_SAVE_PATH}...")
        try:
            global_agent.load_weights(MODEL_SAVE_PATH)
            print("Weights successfully loaded. Resuming training.")
        except Exception as e:
            print(f"Error loading weights ({e}). Starting training from scratch.")

    manager = Manager()
    memory_queue = Queue(maxsize=REPLAY_MEMORY_SIZE) 
    
    shared_weights = manager.list(global_agent.get_weights())
    epsilon_map = manager.dict({'epsilon': 1.0})
    global_steps = manager.dict({'value': 0})
    lock = manager.Lock()

    # 모니터링용 에이전트/환경
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
        
        if memory_queue.qsize() < batch_size * 4: 
            print(f"Waiting for experience... Current size: {memory_queue.qsize()}", end='\r')
            time.sleep(1)
            continue
            
        global_agent.replay(memory_queue, batch_size)
        global_train_count += 1
        
        if global_train_count % target_update_freq == 0:
            global_agent.update_target_model()
        
        # 가중치 공유
        if global_train_count % 1 == 0: 
             new_weights = global_agent.get_weights()
             for i, w in enumerate(new_weights):
                 shared_weights[i] = w
        
        # 주기적 저장 (1000 스텝마다)
        if global_train_count % 1000 == 0 and global_train_count > 0:
            print(f"\n--- Saving model weights at Train Step {global_train_count} ---")
            global_agent.save_weights(MODEL_SAVE_PATH)
        
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
                    monitor_env.close()
                    for p in workers: p.terminate(); p.join()
                    return

        # 로그 출력
        if global_train_count % 10 == 0:
            print(f"Train Step: {global_train_count}/{episodes}, Global Steps: {global_steps['value']}, Epsilon: {epsilon_map['epsilon']:.4f} | Monitor Reward: {monitor_total_reward:.2f}")

    print("\n--- Distributed Training Finished. Saving Model Weights ---")
    global_agent.save_weights(MODEL_SAVE_PATH)
    
    monitor_env.close()
    for p in workers:
        p.terminate()
        p.join()
        
    print("Training Complete.")

if __name__ == '__main__':
    pygame.init() 
    distributed_train_dqn(episodes=50000, batch_size=128, target_update_freq=10, render_freq=5, worker_count=N_WORKERS)