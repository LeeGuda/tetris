import tensorflow as tf
from collections import deque
import numpy as np
import random
import pygame
import os
import time
from multiprocessing import Process, Queue, Manager, cpu_count

AdamOptimizer = tf.keras.optimizers.Adam

# 전역 상수 설정 (모든 프로세스가 공유)
STATE_SIZE = 13
ACTION_MAP = [(r, x) for r in range(4) for x in range(10)]
ACTION_SIZE = len(ACTION_MAP)
REPLAY_MEMORY_SIZE = 20000 
N_WORKERS = cpu_count() - 1 

# 💡 모델 저장 경로 상수 추가
MODEL_SAVE_PATH = 'dqn_tetris_weights.weights.h5' 

class DQNAgent:
    """중앙 및 모니터링 에이전트로 사용되는 클래스"""
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """신경망 모델 구축 (Keras 사용). CPU 장치를 명시적으로 지정합니다."""
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=AdamOptimizer(learning_rate=self.learning_rate))
            return model

    def update_target_model(self):
        """메인 모델의 가중치를 타겟 모델로 복사합니다."""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_weights(self):
        """Worker/Monitor에게 전달할 모델 가중치를 반환합니다."""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """가중치를 받아 모델을 업데이트합니다."""
        self.model.set_weights(weights)
    
    # 💡 가중치 저장 메서드 추가
    def save_weights(self, filename):
        """모델 가중치를 파일로 저장합니다."""
        # CPU 컨텍스트 내에서 저장하여 멀티프로세싱 환경에서 안정성 확보
        with tf.device('/cpu:0'):
            self.model.save_weights(filename)

    # 💡 가중치 로드 메서드 추가
    def load_weights(self, filename):
        """파일에서 모델 가중치를 로드합니다."""
        with tf.device('/cpu:0'):
            self.model.load_weights(filename)
            self.update_target_model() # 타겟 모델도 함께 업데이트

    def act(self, state, possible_actions, ACTION_MAP, epsilon=0.0):
        """행동을 선택합니다."""
        # 1. 탐험 
        if np.random.rand() <= epsilon:
            random_action_tuple = random.choice(possible_actions)
            try:
                action_index = ACTION_MAP.index(random_action_tuple)
            except ValueError:
                action_index = random.randrange(self.action_size)
            return action_index
        
        # 2. 활용 (Exploitation): 최적 행동 선택
        with tf.device('/cpu:0'):
            state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
            q_values_tensor = self.model(state_tensor, training=False)
            q_values = q_values_tensor.numpy()[0]
        
        # 유효한 행동만 고려하여 Q 값 마스킹
        possible_indices = {ACTION_MAP.index(act) for act in possible_actions if act in ACTION_MAP}
        
        for i in range(self.action_size):
            if i not in possible_indices:
                q_values[i] = -1e9  # 유효하지 않은 행동은 무시
                
        action_index = np.argmax(q_values)
        return action_index

    def replay(self, memory_queue, batch_size):
        """공유 메모리에서 미니배치를 샘플링하고 학습합니다."""
        if memory_queue.qsize() < batch_size:
            return

        batch = []
        while not memory_queue.empty() and len(batch) < batch_size:
            batch.append(memory_queue.get())
            
        if not batch: return

        states = np.array([e[0] for e in batch])
        action_indices = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        with tf.device('/cpu:0'):
            next_q_values = self.target_model(next_states, training=False).numpy()
            targets = rewards + self.gamma * np.amax(next_q_values, axis=1) * (1 - dones.astype(int))
            
            target_f = self.model(states, training=False).numpy() 
            
            for i in range(len(batch)):
                target_f[i, action_indices[i]] = targets[i]
            
            self.model.train_on_batch(states, target_f)


def worker_process(worker_id, memory_queue, shared_weights, epsilon_map, global_steps, lock):
    """작업자 프로세스 (Worker Agent)"""
    env = TetrisEnv(render_mode='none') 
    local_agent = DQNAgent()
    
    local_agent.set_weights(shared_weights)
    
    print(f"Worker {worker_id} started. Initial Epsilon: {epsilon_map['epsilon']:.4f}")
    
    while True:
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
        
        with lock:
            if epsilon_map['epsilon'] > 0.01:
                epsilon_map['epsilon'] *= 0.995


def distributed_train_dqn(episodes=50000, batch_size=128, target_update_freq=10, render_freq=5, worker_count=N_WORKERS):
    
    # 1. 중앙 에이전트 및 공유 자원 설정
    global_agent = DQNAgent() 
    
    # 💡 저장된 모델 가중치 로드
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading previous weights from {MODEL_SAVE_PATH}...")
        try:
            global_agent.load_weights(MODEL_SAVE_PATH)
            print("Weights successfully loaded. Resuming training.")
        except Exception as e:
            # 모델 구조가 변경되었거나 파일이 손상된 경우
            print(f"Error loading weights ({e}). Starting training from scratch.")

    manager = Manager()
    memory_queue = Queue(maxsize=REPLAY_MEMORY_SIZE) 
    
    shared_weights = manager.list(global_agent.get_weights())
    epsilon_map = manager.dict({'epsilon': 1.0})
    global_steps = manager.dict({'value': 0})
    lock = manager.Lock()

    # 2. 렌더링을 위한 별도 모니터링 에이전트 및 환경 생성
    monitor_agent = DQNAgent() 
    monitor_env = TetrisEnv(render_mode='human')
    monitor_state = monitor_env.reset()
    monitor_done = False
    
    monitor_total_reward = 0.0
    monitor_step_count = 0
    
    # 3. Worker 프로세스 생성 및 시작
    print(f"\n--- Starting Distributed Training with {worker_count} Workers (CPU Mode) ---")
    workers = []
    actual_worker_count = max(1, worker_count)
    for i in range(actual_worker_count):
        p = Process(target=worker_process, args=(i, memory_queue, shared_weights, epsilon_map, global_steps, lock))
        workers.append(p)
        p.start()

    # 4. 중앙 학습 루프 (메인 프로세스)
    global_train_count = 0
    total_steps = 0
    
    while global_train_count < episodes:
        
        if memory_queue.qsize() < batch_size * 4: 
            print(f"Waiting for experience... Current size: {memory_queue.qsize()}", end='\r')
            time.sleep(1)
            continue
            
        # 4.1 모델 학습
        global_agent.replay(memory_queue, batch_size)
        global_train_count += 1
        
        # 4.2 타겟 모델 업데이트
        if global_train_count % target_update_freq == 0:
            global_agent.update_target_model()
        
        # 4.3 Worker들에게 업데이트된 가중치 동기화
        if global_train_count % 1 == 0: 
             new_weights = global_agent.get_weights()
             for i, w in enumerate(new_weights):
                 shared_weights[i] = w
        
        # 💡 4.4 주기적인 모델 저장 (1000 학습 스텝마다)
        if global_train_count % 1000 == 0 and global_train_count > 0:
            print(f"\n--- Saving model weights at Train Step {global_train_count} ---")
            global_agent.save_weights(MODEL_SAVE_PATH)
        
        # 4.5 주기적인 렌더링 및 모니터링 (콘솔 출력용)
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
            
            # TetrisEnv.py의 render 함수는 인수를 받지 않도록 수정되어야 합니다.
            monitor_env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nUser quit signal received. Terminating training and saving model...")
                    # 💡 종료 시 최종 가중치 저장
                    global_agent.save_weights(MODEL_SAVE_PATH) 
                    monitor_env.close()
                    for p in workers: p.terminate(); p.join()
                    return

        # 4.6 통계 출력 (콘솔)
        total_steps = global_steps['value']
        if global_train_count % 10 == 0:
            print(f"Train Step: {global_train_count}/{episodes}, Steps: {total_steps}, Epsilon: {epsilon_map['epsilon']:.4f} | Monitor -> Steps: {monitor_step_count}, Reward: {monitor_total_reward:.2f}")

    # 5. 최종 종료
    print("\n--- Distributed Training Finished. Finalizing and Saving Model Weights ---")
    global_agent.save_weights(MODEL_SAVE_PATH) # 최종 가중치 저장
    
    monitor_env.close()
    for p in workers:
        p.terminate()
        p.join()
        
    print("Training Complete.")

if __name__ == '__main__':
    pygame.init() 
    distributed_train_dqn(episodes=50000, batch_size=128, target_update_freq=10, render_freq=5, worker_count=N_WORKERS)