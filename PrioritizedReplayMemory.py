import numpy as np
import random
import os
import pickle

class PrioritizedReplayMemory:
    """우선순위 기반 리플레이 메모리 (Segment Tree 사용)"""
    
    # PER 하이퍼파라미터
    e = 0.01  # TD 오차에 더해주는 작은 값 (오차가 0인 경험도 샘플링되게 함)
    a = 0.6   # 우선순위의 영향을 결정하는 지수 (0: 일반 DQN, 1: 완전 우선순위)
    beta = 0.4  # 중요도 샘플링 가중치 (IS weight) 보정 지수
    beta_increment_per_sampling = 0.001 # beta 값 증가 스텝

    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.state_size = state_size
        self.memory = []
        self.idx = 0  # 다음 저장 위치 인덱스
        
        # Segment Tree 크기: 2의 거듭제곱으로 capacity보다 크거나 같은 최소값
        self.tree_size = 1
        while self.tree_size < self.capacity:
            self.tree_size *= 2
            
        # Segment Tree 초기화 (총합과 최소값 저장용)
        self.sum_tree = np.zeros(2 * self.tree_size)
        self.min_tree = np.full(2 * self.tree_size, float('inf')) 
        
        # PER 관련 변수
        self.max_priority = 1.0 # 초기 최대 우선순위
        
    # --- Segment Tree 유틸리티 함수 ---
    def _update_tree(self, tree, idx, value):
        idx += self.tree_size
        tree[idx] = value
        
        while idx > 1:
            idx //= 2
            tree[idx] = tree[2 * idx] + tree[2 * idx + 1] if tree is self.sum_tree else min(tree[2 * idx], tree[2 * idx + 1])
            
    def _retrieve(self, b):
        idx = 1
        while idx < self.tree_size:
            if self.sum_tree[2 * idx] < b:
                b -= self.sum_tree[2 * idx]
                idx = 2 * idx + 1
            else:
                idx = 2 * idx
        return idx - self.tree_size # 실제 메모리 인덱스 반환

    def _get_prefix_sum(self, start, end):
        # 쿼리 범위에 대한 합계를 반환
        start += self.tree_size
        end += self.tree_size
        s = 0
        while start <= end:
            if start % 2 == 1:
                s += self.sum_tree[start]
                start += 1
            if end % 2 == 0:
                s += self.sum_tree[end]
                end -= 1
            start //= 2
            end //= 2
        return s
        
    # --- 핵심 메모리 기능 ---
    
    def store(self, experience, priority):
        """경험을 저장하고 우선순위를 업데이트합니다."""
        
        # 1. 메모리 데이터 저장
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.idx] = experience
            
        # 2. Segment Tree에 우선순위 업데이트
        p = priority ** self.a
        self._update_tree(self.sum_tree, self.idx, p)
        self._update_tree(self.min_tree, self.idx, p)
        
        # 3. 인덱스 업데이트
        self.idx = (self.idx + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)
        
    def sample(self, batch_size):
        """우선순위에 따라 경험을 샘플링하고 IS 가중치를 반환합니다."""
        
        batch = []
        indices = []
        is_weights = np.empty((batch_size, 1), dtype=np.float32)
        
        total_priority_sum = self.sum_tree[1]
        
        # IS 가중치 보정 계수 beta 증가
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        # 최소 우선순위 (IS weight 계산용)
        min_p = self.min_tree[1]
        
        # total_priority_sum / min_p = 최대 중요도 가중치 (W_max)
        # IS_weight = (N * P(i))^-beta / W_max 
        # P(i) = p_i / total_priority_sum
        
        # batch_size만큼 샘플링
        for i in range(batch_size):
            # 1. 샘플링 값 (prefix sum) 범위 지정
            segment = total_priority_sum / batch_size
            b = random.uniform(segment * i, segment * (i + 1))
            
            # 2. 우선순위 값에 해당하는 인덱스 찾기
            memory_idx = self._retrieve(b)
            
            # 3. 우선순위 값과 확률 계산
            p_i = self.sum_tree[memory_idx + self.tree_size]
            prob_i = p_i / total_priority_sum
            
            # 4. IS Weight 계산 (분모에 capacity 대신 실제 메모리 크기 사용)
            is_weights[i] = (len(self.memory) * prob_i) ** (-self.beta)
            
            indices.append(memory_idx)
            batch.append(self.memory[memory_idx])
        
        # IS Weight 정규화 (최대값으로 나누어 스케일링)
        max_is_weight = is_weights.max()
        is_weights /= max_is_weight if max_is_weight > 0 else 1.0

        # 배열로 변환
        states = np.array([e[0] for e in batch])
        action_indices = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, action_indices, rewards, next_states, dones, indices, is_weights

    def update_priorities(self, indices, td_errors):
        """TD 오차를 기반으로 우선순위 트리를 업데이트합니다."""
        for idx, error in zip(indices, td_errors):
            priority = (np.abs(error) + self.e)
            p = priority ** self.a
            self.max_priority = max(self.max_priority, priority)
            self._update_tree(self.sum_tree, idx, p)
            self._update_tree(self.min_tree, idx, p)

    def size(self):
        return len(self.memory)

    # --- 저장 및 로드 기능 ---
    def save_memory(self, filename):
        """메모리 객체를 파일로 저장합니다."""
        data = {
            'memory': self.memory,
            'idx': self.idx,
            'sum_tree': self.sum_tree,
            'min_tree': self.min_tree,
            'max_priority': self.max_priority,
            'beta': self.beta,
            'capacity': self.capacity,
            'state_size': self.state_size
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[PER] Memory saved to {filename}. Current size: {len(self.memory)}")

    def load_memory(self, filename):
        """파일에서 메모리 객체를 불러옵니다."""
        if not os.path.exists(filename):
            print(f"[PER] Memory file {filename} not found. Starting fresh memory.")
            return

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if data['capacity'] != self.capacity:
            print(f"[PER] WARNING: Loaded capacity ({data['capacity']}) mismatch with current capacity ({self.capacity}). Resizing.")
            # 용량 불일치 시 기존 데이터를 자르거나 확장하는 로직이 필요하지만, 여기서는 단순 로드
            pass 
            
        self.memory = data['memory']
        self.idx = data['idx']
        self.sum_tree = data['sum_tree']
        self.min_tree = data['min_tree']
        self.max_priority = data['max_priority']
        self.beta = data['beta']
        self.capacity = data['capacity']
        self.state_size = data['state_size']
        
        # Segment Tree는 capacity를 기반으로 생성되므로, 여기서 tree_size를 다시 계산해야 함
        self.tree_size = 1
        while self.tree_size < self.capacity:
            self.tree_size *= 2
        
        print(f"[PER] Memory loaded from {filename}. Current size: {len(self.memory)}/{self.capacity}")